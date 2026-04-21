from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import torch
from transformers import TimesFmModelForPrediction

from src.data.datasets import load_dataset
from src.eval.geometry_metrics import (
    compute_css,
    compute_fss,
    compute_lls,
    compute_ltc,
    compute_mcs,
    compute_pds,
    compute_tcs,
    primitive_labels_from_stl_features,
)
from src.eval.metrics import compute_gsp, compute_lcs, compute_tsi
from src.eval.perturbation import apply_masking, apply_noise, apply_time_warp
from src.eval.regime_labels import phase_mod_labels, stl_regime_labels
from src.latents.visualize import project_umap, project_umap_3d, save_latent_scatter3d_html, save_scatter
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TimesFM latent extraction demo")
    parser.add_argument("--dataset", type=str, default="electricity")
    parser.add_argument("--model-id", type=str, default="google/timesfm-2.0-500m-pytorch")
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--max-windows", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--weight-mode",
        type=str,
        default="both",
        choices=["pretrained", "random-reset", "both"],
        help="Use pretrained TimesFM, random-reset control, or run both.",
    )
    parser.add_argument(
        "--label-mode",
        type=str,
        default="stl",
        choices=["phase8", "stl", "both"],
        help="Use legacy mod-8 labels, STL-derived labels, or evaluate both.",
    )
    parser.add_argument("--stl-period", type=int, default=24, help="Seasonality period used by STL.")
    parser.add_argument("--stl-clusters", type=int, default=8, help="Number of clusters for STL-feature regimes.")
    parser.add_argument(
        "--stl-periods",
        type=str,
        default="",
        help="Comma list for STL period sweep, e.g. '24,48,96'. Empty => use --stl-period only.",
    )
    parser.add_argument(
        "--stl-clusters-list",
        type=str,
        default="",
        help="Comma list for STL cluster sweep, e.g. '4,8,12'. Empty => use --stl-clusters only.",
    )
    parser.add_argument(
        "--noise-scales",
        type=str,
        default="0.02",
        help="Comma list for AddNoise scale sweep.",
    )
    parser.add_argument(
        "--mask-ps",
        type=str,
        default="0.05",
        help="Comma list for Dropout p sweep.",
    )
    parser.add_argument(
        "--warp-ratios",
        type=str,
        default="2.0",
        help="Comma list for TimeWarp max_speed_ratio sweep.",
    )
    parser.add_argument("--warp-speed-changes", type=int, default=3, help="TimeWarp n_speed_change.")
    parser.add_argument(
        "--horizons",
        type=str,
        default="96,192,336,720",
        help="Comma list of forecast horizons for phase-1 geometry metrics.",
    )
    parser.add_argument("--phase1-knn", type=str, default="10,20,50", help="Comma list of k for FSS/PDS k-NN.")
    parser.add_argument("--bootstrap-resamples", type=int, default=500, help="Bootstrap resamples for CI.")
    parser.add_argument("--mcs-quantile", type=float, default=0.95, help="Support quantile for MCS.")
    parser.add_argument("--css-permutations", type=int, default=100, help="Permutation count for CSS p-value.")
    parser.add_argument("--fast-dev", action="store_true", help="Reduce expensive computations for quick iteration.")
    parser.add_argument(
        "--fss-max-pairs",
        type=int,
        default=20000,
        help="Max random pairs for FSS computation to control runtime.",
    )
    return parser.parse_args()


def build_windows(
    series: np.ndarray, context_length: int, max_windows: int, max_horizon: int = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(series)
    if n <= context_length + max_horizon:
        raise ValueError("Series too short for requested context length + max horizon")
    starts = np.linspace(0, n - context_length - max_horizon, num=max_windows, dtype=int)
    x = np.stack([series[s : s + context_length] for s in starts], axis=0).astype(np.float32)
    y = np.stack([series[s + context_length : s + context_length + max_horizon] for s in starts], axis=0).astype(
        np.float32
    )
    return x, starts, y


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _reset_module_parameters(module: torch.nn.Module) -> None:
    for layer in module.modules():
        if hasattr(layer, "reset_parameters") and callable(layer.reset_parameters):
            layer.reset_parameters()


def load_model(model_id: str, device: torch.device, mode: str, seed: int) -> TimesFmModelForPrediction:
    model = TimesFmModelForPrediction.from_pretrained(model_id)
    if mode == "random-reset":
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        _reset_module_parameters(model)
    return model.to(device).eval()


def _label_sets(args: argparse.Namespace, windows: np.ndarray, n_windows: int, seed: int) -> dict[str, np.ndarray]:
    modes = [args.label_mode] if args.label_mode != "both" else ["phase8", "stl"]
    out: dict[str, np.ndarray] = {}
    if "phase8" in modes:
        out["phase8::na"] = phase_mod_labels(n_windows, modulus=8)
    if "stl" in modes:
        periods = _parse_int_list(args.stl_periods, fallback=[args.stl_period], min_value=2)
        clusters = _parse_int_list(args.stl_clusters_list, fallback=[args.stl_clusters], min_value=2)
        for period, k in itertools.product(periods, clusters):
            labels, _ = stl_regime_labels(
                windows=windows,
                period=period,
                n_clusters=k,
                random_state=seed,
            )
            out[f"stl::p{period}_k{k}"] = labels
    return out


def _extract_sequence_embeddings(
    model: TimesFmModelForPrediction,
    past_values: torch.Tensor,
    freq: torch.Tensor,
) -> np.ndarray:
    with torch.no_grad():
        outputs = model(
            past_values=past_values,
            freq=freq,
            output_hidden_states=True,
            return_dict=True,
        )
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None or len(hidden_states) == 0:
        raise RuntimeError("TimesFM output did not contain hidden_states.")
    last_hidden = hidden_states[-1].detach().float().cpu().numpy()  # [B, T, D]
    return last_hidden.mean(axis=1)  # [B, D]


def _perturb_past_values(past_values_np: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = past_values_np[:, :, None]  # [B, T, 1] so perturbation runs on time axis T.
    noisy = apply_noise(x).squeeze(-1)
    masked = apply_masking(x).squeeze(-1)
    warped = apply_time_warp(x).squeeze(-1)
    return noisy, masked, warped


def _parse_float_list(raw: str, fallback: list[float]) -> list[float]:
    if raw.strip() == "":
        return fallback
    values = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not values:
        return fallback
    return values


def _parse_int_list(raw: str, fallback: list[int], min_value: int = 0) -> list[int]:
    if raw.strip() == "":
        return fallback
    values = [int(x.strip()) for x in raw.split(",") if x.strip()]
    values = [v for v in values if v >= min_value]
    if not values:
        return fallback
    return values


def _time_resample_same_length(x: np.ndarray, speed: float) -> np.ndarray:
    # Resample on a warped time grid then interpolate back to original length.
    b, t = x.shape
    base_t = np.linspace(0.0, 1.0, num=t, dtype=np.float32)
    warped_t = np.clip(base_t**speed, 0.0, 1.0)
    out = np.empty_like(x, dtype=np.float32)
    for i in range(b):
        out[i] = np.interp(base_t, warped_t, x[i]).astype(np.float32)
    return out


def _summarize_metrics(metrics_df):
    import pandas as pd

    if metrics_df.empty:
        return pd.DataFrame()
    grouped = (
        metrics_df.groupby(["dataset", "weight_mode", "label_mode", "metric"], as_index=False)["value"]
        .agg(best="max", median="median", mean="mean", worst="min")
    )
    return grouped


def _forecast_normalized(
    model: TimesFmModelForPrediction,
    past_values: torch.Tensor,
    freq: torch.Tensor,
    horizon: int,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    current = past_values
    while sum(chunk.shape[1] for chunk in chunks) < horizon:
        with torch.no_grad():
            outputs = model(
                past_values=current,
                freq=freq,
                output_hidden_states=False,
            )
        pred_chunk = outputs.mean_predictions.detach().float().cpu().numpy()
        chunks.append(pred_chunk)
        current_np = np.concatenate([current.detach().float().cpu().numpy(), pred_chunk], axis=1)
        current = torch.from_numpy(current_np.astype(np.float32)).to(past_values.device)
    pred = np.concatenate(chunks, axis=1)[:, :horizon]
    return pred


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device()

    print(f"[timesfm] loading dataset={args.dataset}")
    ds = load_dataset(args.dataset)
    series = ds.frame[ds.target_col].astype(float).to_numpy()

    horizons = sorted(_parse_int_list(args.horizons, fallback=[96, 192, 336, 720], min_value=1))
    max_h = int(max(horizons))
    windows, starts, future_targets = build_windows(
        series,
        context_length=args.context_length,
        max_windows=args.max_windows,
        max_horizon=max_h,
    )
    # Fold-specific normalization for stable inference.
    mu = windows.mean(axis=1, keepdims=True)
    sigma = windows.std(axis=1, keepdims=True) + 1e-6
    windows_norm = (windows - mu) / sigma

    print(f"[timesfm] model={args.model_id} device={device} windows={len(windows_norm)}")
    weight_modes = [args.weight_mode] if args.weight_mode != "both" else ["pretrained", "random-reset"]

    past_values = torch.from_numpy(windows_norm).to(device)
    freq = torch.zeros((past_values.shape[0],), dtype=torch.long, device=device)
    label_sets = _label_sets(args, windows=windows, n_windows=len(windows), seed=args.seed)
    knn_values = sorted(_parse_int_list(args.phase1_knn, fallback=[10, 20, 50], min_value=2))
    noise_scales = _parse_float_list(args.noise_scales, fallback=[0.02])
    mask_ps = _parse_float_list(args.mask_ps, fallback=[0.05])
    warp_ratios = _parse_float_list(args.warp_ratios, fallback=[2.0])
    perturb_grid = list(itertools.product(noise_scales, mask_ps, warp_ratios))

    out_dir = Path("outputs/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = Path("outputs/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows: list[dict[str, float | int | str]] = []
    for weight_mode in weight_modes:
        print(f"[timesfm] running mode={weight_mode}")
        model = load_model(args.model_id, device=device, mode=weight_mode, seed=args.seed)

        embeddings = _extract_sequence_embeddings(model=model, past_values=past_values, freq=freq)
        pred_norm = _forecast_normalized(model=model, past_values=past_values, freq=freq, horizon=max_h)
        pred = pred_norm * sigma + mu
        abs_err = np.abs(pred - future_targets)
        sq_err = (pred - future_targets) ** 2
        time_idx = starts.astype(float)

        coords2 = project_umap(embeddings, random_state=args.seed)
        png_path = out_dir / f"figure_timesfm_umap_{args.dataset}_{weight_mode}.png"
        save_scatter(coords2, time_idx, png_path, title=f"TimesFM latent 2D ({args.dataset}, {weight_mode})")

        coords3 = project_umap_3d(embeddings, random_state=args.seed)
        html_path = out_dir / f"figure_timesfm_3d_{args.dataset}_{weight_mode}.html"
        save_latent_scatter3d_html(
            coords3,
            time_idx,
            html_path,
            title=f"TimesFM latent 3D ({args.dataset}, {weight_mode})",
            subtitle="Hidden-state sequence mean per window",
            max_points=2000,
            trajectory_max_points=600,
            include_plotlyjs=True,
        )

        for noise_scale, mask_p, warp_ratio in perturb_grid:
            x = windows_norm[:, :, None]
            noisy_pv = apply_noise(x, scale=noise_scale).squeeze(-1)
            masked_pv = apply_masking(x, p=mask_p).squeeze(-1)
            warped_pv = apply_time_warp(
                x,
                n_speed_change=args.warp_speed_changes,
                max_speed_ratio=warp_ratio,
            ).squeeze(-1)

            noisy_emb = _extract_sequence_embeddings(
                model=model,
                past_values=torch.from_numpy(noisy_pv.astype(np.float32)).to(device),
                freq=freq,
            )
            masked_emb = _extract_sequence_embeddings(
                model=model,
                past_values=torch.from_numpy(masked_pv.astype(np.float32)).to(device),
                freq=freq,
            )
            warped_emb = _extract_sequence_embeddings(
                model=model,
                past_values=torch.from_numpy(warped_pv.astype(np.float32)).to(device),
                freq=freq,
            )

            perturb_id = f"n{noise_scale:g}_m{mask_p:g}_w{warp_ratio:g}"
            for label_key, labels in label_sets.items():
                label_mode, stl_setting = label_key.split("::", maxsplit=1)
                lcs_clean = compute_lcs(embeddings, labels)
                lcs_noisy = compute_lcs(noisy_emb, labels)
                lcs_masked = compute_lcs(masked_emb, labels)
                lcs_warped = compute_lcs(warped_emb, labels)

                tsi_noisy = compute_tsi(embeddings, noisy_emb)
                tsi_masked = compute_tsi(embeddings, masked_emb)
                tsi_warped = compute_tsi(embeddings, warped_emb)

                metrics_rows.extend(
                    [
                        {
                            "dataset": args.dataset,
                            "weight_mode": weight_mode,
                            "label_mode": label_mode,
                            "stl_setting": stl_setting,
                            "perturb_setting": perturb_id,
                            "noise_scale": noise_scale,
                            "mask_p": mask_p,
                            "warp_ratio": warp_ratio,
                            "metric": "LCS_clean",
                            "value": lcs_clean,
                        },
                        {
                            "dataset": args.dataset,
                            "weight_mode": weight_mode,
                            "label_mode": label_mode,
                            "stl_setting": stl_setting,
                            "perturb_setting": perturb_id,
                            "noise_scale": noise_scale,
                            "mask_p": mask_p,
                            "warp_ratio": warp_ratio,
                            "metric": "LCS_noise",
                            "value": lcs_noisy,
                        },
                        {
                            "dataset": args.dataset,
                            "weight_mode": weight_mode,
                            "label_mode": label_mode,
                            "stl_setting": stl_setting,
                            "perturb_setting": perturb_id,
                            "noise_scale": noise_scale,
                            "mask_p": mask_p,
                            "warp_ratio": warp_ratio,
                            "metric": "LCS_mask",
                            "value": lcs_masked,
                        },
                        {
                            "dataset": args.dataset,
                            "weight_mode": weight_mode,
                            "label_mode": label_mode,
                            "stl_setting": stl_setting,
                            "perturb_setting": perturb_id,
                            "noise_scale": noise_scale,
                            "mask_p": mask_p,
                            "warp_ratio": warp_ratio,
                            "metric": "LCS_warp",
                            "value": lcs_warped,
                        },
                        {
                            "dataset": args.dataset,
                            "weight_mode": weight_mode,
                            "label_mode": label_mode,
                            "stl_setting": stl_setting,
                            "perturb_setting": perturb_id,
                            "noise_scale": noise_scale,
                            "mask_p": mask_p,
                            "warp_ratio": warp_ratio,
                            "metric": "TSI_noise",
                            "value": tsi_noisy,
                        },
                        {
                            "dataset": args.dataset,
                            "weight_mode": weight_mode,
                            "label_mode": label_mode,
                            "stl_setting": stl_setting,
                            "perturb_setting": perturb_id,
                            "noise_scale": noise_scale,
                            "mask_p": mask_p,
                            "warp_ratio": warp_ratio,
                            "metric": "TSI_mask",
                            "value": tsi_masked,
                        },
                        {
                            "dataset": args.dataset,
                            "weight_mode": weight_mode,
                            "label_mode": label_mode,
                            "stl_setting": stl_setting,
                            "perturb_setting": perturb_id,
                            "noise_scale": noise_scale,
                            "mask_p": mask_p,
                            "warp_ratio": warp_ratio,
                            "metric": "TSI_warp",
                            "value": tsi_warped,
                        },
                        {
                            "dataset": args.dataset,
                            "weight_mode": weight_mode,
                            "label_mode": label_mode,
                            "stl_setting": stl_setting,
                            "perturb_setting": perturb_id,
                            "noise_scale": noise_scale,
                            "mask_p": mask_p,
                            "warp_ratio": warp_ratio,
                            "metric": "GSP_noise",
                            "value": compute_gsp(lcs_clean, lcs_noisy),
                        },
                        {
                            "dataset": args.dataset,
                            "weight_mode": weight_mode,
                            "label_mode": label_mode,
                            "stl_setting": stl_setting,
                            "perturb_setting": perturb_id,
                            "noise_scale": noise_scale,
                            "mask_p": mask_p,
                            "warp_ratio": warp_ratio,
                            "metric": "GSP_mask",
                            "value": compute_gsp(lcs_clean, lcs_masked),
                        },
                        {
                            "dataset": args.dataset,
                            "weight_mode": weight_mode,
                            "label_mode": label_mode,
                            "stl_setting": stl_setting,
                            "perturb_setting": perturb_id,
                            "noise_scale": noise_scale,
                            "mask_p": mask_p,
                            "warp_ratio": warp_ratio,
                            "metric": "GSP_warp",
                            "value": compute_gsp(lcs_clean, lcs_warped),
                        },
                    ]
                )

        # Geometry metrics (clean embeddings): FSS, PDS, LTC + phase-2 (LLS/MCS/CSS/TCS).
        ltc_step, ltc_dir = compute_ltc(embeddings=embeddings, starts=starts)
        metrics_rows.extend(
            [
                {
                    "dataset": args.dataset,
                    "weight_mode": weight_mode,
                    "label_mode": "na",
                    "stl_setting": "na",
                    "perturb_setting": "clean",
                    "noise_scale": 0.0,
                    "mask_p": 0.0,
                    "warp_ratio": 0.0,
                    "metric": "LTC_step",
                    "value": ltc_step,
                },
                {
                    "dataset": args.dataset,
                    "weight_mode": weight_mode,
                    "label_mode": "na",
                    "stl_setting": "na",
                    "perturb_setting": "clean",
                    "noise_scale": 0.0,
                    "mask_p": 0.0,
                    "warp_ratio": 0.0,
                    "metric": "LTC_dir",
                    "value": ltc_dir,
                },
            ]
        )

        horizon_slices: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
        for h in horizons:
            horizon_slices.append((str(h), pred[:, h - 1], abs_err[:, h - 1], sq_err[:, h - 1]))
        horizon_slices.append(("avg", pred[:, :max_h].mean(axis=1), abs_err[:, :max_h].mean(axis=1), sq_err[:, :max_h].mean(axis=1)))

        for h_key, pred_h, mae_h, mse_h in horizon_slices:
            h_seed = max_h if h_key == "avg" else int(h_key)
            for corr_type in ["spearman", "pearson"]:
                fss_stat, fss_low, fss_high, fss_p, fss_raw_corr = compute_fss(
                    embeddings=embeddings,
                    preds=pred_h,
                    corr_type=corr_type,
                    max_pairs=args.fss_max_pairs,
                    n_resamples=args.bootstrap_resamples,
                    seed=args.seed + h_seed,
                )
                for suffix, value in [
                    ("score", fss_stat),
                    ("raw_corr", fss_raw_corr),
                    ("ci_low", fss_low),
                    ("ci_high", fss_high),
                    ("pvalue", fss_p),
                ]:
                    metrics_rows.append(
                        {
                            "dataset": args.dataset,
                            "weight_mode": weight_mode,
                            "label_mode": "na",
                            "stl_setting": "na",
                            "perturb_setting": "clean",
                            "noise_scale": 0.0,
                            "mask_p": 0.0,
                            "warp_ratio": 0.0,
                            "metric": f"FSS_{corr_type}_h{h_key}_{suffix}",
                            "value": float(value),
                        }
                    )

                for k in knn_values:
                    for err_name, err_vec in [("mae", mae_h), ("mse", mse_h)]:
                        pds_stat, pds_low, pds_high, pds_p, pds_raw_corr = compute_pds(
                            embeddings=embeddings,
                            errors=err_vec,
                            k=k,
                            corr_type=corr_type,
                            n_resamples=args.bootstrap_resamples,
                            seed=args.seed + h_seed + k,
                        )
                        for suffix, value in [
                            ("score", pds_stat),
                            ("raw_corr", pds_raw_corr),
                            ("ci_low", pds_low),
                            ("ci_high", pds_high),
                            ("pvalue", pds_p),
                        ]:
                            metrics_rows.append(
                                {
                                    "dataset": args.dataset,
                                    "weight_mode": weight_mode,
                                    "label_mode": "na",
                                    "stl_setting": "na",
                                    "perturb_setting": f"clean_k{k}",
                                    "noise_scale": 0.0,
                                    "mask_p": 0.0,
                                    "warp_ratio": 0.0,
                                    "metric": f"PDS_{corr_type}_{err_name}_h{h_key}_{suffix}",
                                    "value": float(value),
                                }
                            )

        # Phase-2 metrics.
        lls_vals = {k: compute_lls(embeddings, k=k) for k in knn_values}
        lls_avg = float(np.mean(list(lls_vals.values()))) if lls_vals else 0.0
        for k, value in lls_vals.items():
            metrics_rows.append(
                {
                    "dataset": args.dataset,
                    "weight_mode": weight_mode,
                    "label_mode": "na",
                    "stl_setting": "na",
                    "perturb_setting": "clean",
                    "noise_scale": 0.0,
                    "mask_p": 0.0,
                    "warp_ratio": 0.0,
                    "metric": f"LLS_k{k}",
                    "value": float(value),
                }
            )
        metrics_rows.append(
            {
                "dataset": args.dataset,
                "weight_mode": weight_mode,
                "label_mode": "na",
                "stl_setting": "na",
                "perturb_setting": "clean",
                "noise_scale": 0.0,
                "mask_p": 0.0,
                "warp_ratio": 0.0,
                "metric": "LLS_avgk",
                "value": lls_avg,
            }
        )

        split = int(0.8 * len(embeddings))
        split = max(1, min(split, len(embeddings) - 1))
        train_emb, test_emb = embeddings[:split], embeddings[split:]
        for k in knn_values:
            mcs = compute_mcs(train_embeddings=train_emb, test_embeddings=test_emb, k=k, quantile=args.mcs_quantile)
            metrics_rows.append(
                {
                    "dataset": args.dataset,
                    "weight_mode": weight_mode,
                    "label_mode": "na",
                    "stl_setting": f"q{args.mcs_quantile:g}",
                    "perturb_setting": "clean",
                    "noise_scale": 0.0,
                    "mask_p": 0.0,
                    "warp_ratio": 0.0,
                    "metric": f"MCS_k{k}",
                    "value": mcs,
                }
            )

        # Primitive probes for CSS from STL features.
        _, stl_feats = stl_regime_labels(windows=windows, period=args.stl_period, n_clusters=args.stl_clusters, random_state=args.seed)
        primitive_labels = primitive_labels_from_stl_features(stl_feats)
        n_perm = 20 if args.fast_dev else args.css_permutations
        for primitive_name, primitive_y in primitive_labels.items():
            css_bal_acc, css_margin, css_pvalue = compute_css(
                embeddings=embeddings,
                labels=primitive_y,
                seed=args.seed,
                n_permutations=n_perm,
            )
            for metric_name, metric_value in [
                (f"CSS_{primitive_name}_bal_acc", css_bal_acc),
                (f"CSS_{primitive_name}_margin", css_margin),
                (f"CSS_{primitive_name}_pvalue", css_pvalue),
            ]:
                metrics_rows.append(
                    {
                        "dataset": args.dataset,
                        "weight_mode": weight_mode,
                        "label_mode": "na",
                        "stl_setting": "na",
                        "perturb_setting": "clean",
                        "noise_scale": 0.0,
                        "mask_p": 0.0,
                        "warp_ratio": 0.0,
                        "metric": metric_name,
                        "value": float(metric_value),
                    }
                )

        # TCS using deterministic transforms.
        transformed_inputs = {
            "scale": (windows_norm * 1.1).astype(np.float32),
            "shift": (windows_norm + 0.1).astype(np.float32),
            "resample": _time_resample_same_length(windows_norm, speed=0.85),
        }
        tcs_values = []
        for t_name, t_values in transformed_inputs.items():
            t_emb = _extract_sequence_embeddings(
                model=model,
                past_values=torch.from_numpy(t_values).to(device),
                freq=freq,
            )
            tcs = compute_tcs(embeddings, t_emb)
            tcs_values.append(tcs)
            metrics_rows.append(
                {
                    "dataset": args.dataset,
                    "weight_mode": weight_mode,
                    "label_mode": "na",
                    "stl_setting": "na",
                    "perturb_setting": "clean",
                    "noise_scale": 0.0,
                    "mask_p": 0.0,
                    "warp_ratio": 0.0,
                    "metric": f"TCS_{t_name}",
                    "value": float(tcs),
                }
            )
        metrics_rows.append(
            {
                "dataset": args.dataset,
                "weight_mode": weight_mode,
                "label_mode": "na",
                "stl_setting": "na",
                "perturb_setting": "clean",
                "noise_scale": 0.0,
                "mask_p": 0.0,
                "warp_ratio": 0.0,
                "metric": "TCS_avg",
                "value": float(np.mean(tcs_values)) if tcs_values else 0.0,
            }
        )

        print(f"[timesfm] saved 2D plot: {png_path}")
        print(f"[timesfm] saved 3D html: {html_path}")

    import pandas as pd

    metrics_df = pd.DataFrame(metrics_rows)
    dedup_keys = [
        "dataset",
        "weight_mode",
        "label_mode",
        "stl_setting",
        "perturb_setting",
        "noise_scale",
        "mask_p",
        "warp_ratio",
        "metric",
    ]
    metrics_df = metrics_df.drop_duplicates(subset=dedup_keys, keep="last")
    metrics_path = metrics_dir / f"timesfm_control_metrics_{args.dataset}.csv"
    metrics_df.to_csv(metrics_path, index=False)
    geometry_metrics_path = metrics_dir / f"timesfm_geometry_metrics_{args.dataset}.csv"
    metrics_df.to_csv(geometry_metrics_path, index=False)
    print(f"[timesfm] saved control metrics: {metrics_path}")
    summary_df = _summarize_metrics(metrics_df)
    summary_path = metrics_dir / f"timesfm_control_summary_{args.dataset}.csv"
    summary_df.to_csv(summary_path, index=False)
    geometry_summary_path = metrics_dir / f"timesfm_geometry_summary_{args.dataset}.csv"
    summary_df.to_csv(geometry_summary_path, index=False)
    print(f"[timesfm] saved control summary (best/median): {summary_path}")


if __name__ == "__main__":
    main()
