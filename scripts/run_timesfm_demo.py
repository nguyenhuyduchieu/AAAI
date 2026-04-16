from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import torch
from transformers import TimesFmModelForPrediction

from src.data.datasets import load_dataset
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
    return parser.parse_args()


def build_windows(series: np.ndarray, context_length: int, max_windows: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(series)
    if n <= context_length:
        raise ValueError("Series too short for requested context length")
    starts = np.linspace(0, n - context_length - 1, num=max_windows, dtype=int)
    x = np.stack([series[s : s + context_length] for s in starts], axis=0).astype(np.float32)
    return x, starts


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


def _summarize_metrics(metrics_df):
    import pandas as pd

    if metrics_df.empty:
        return pd.DataFrame()
    grouped = (
        metrics_df.groupby(["dataset", "weight_mode", "label_mode", "metric"], as_index=False)["value"]
        .agg(best="max", median="median", mean="mean", worst="min")
    )
    return grouped


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device()

    print(f"[timesfm] loading dataset={args.dataset}")
    ds = load_dataset(args.dataset)
    series = ds.frame[ds.target_col].astype(float).to_numpy()

    windows, starts = build_windows(series, context_length=args.context_length, max_windows=args.max_windows)
    # Fold-specific normalization for stable inference.
    mu = windows.mean(axis=1, keepdims=True)
    sigma = windows.std(axis=1, keepdims=True) + 1e-6
    windows_norm = (windows - mu) / sigma

    print(f"[timesfm] model={args.model_id} device={device} windows={len(windows_norm)}")
    weight_modes = [args.weight_mode] if args.weight_mode != "both" else ["pretrained", "random-reset"]

    past_values = torch.from_numpy(windows_norm).to(device)
    freq = torch.zeros((past_values.shape[0],), dtype=torch.long, device=device)
    label_sets = _label_sets(args, windows=windows, n_windows=len(windows), seed=args.seed)
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

        print(f"[timesfm] saved 2D plot: {png_path}")
        print(f"[timesfm] saved 3D html: {html_path}")

    import pandas as pd

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = metrics_dir / f"timesfm_control_metrics_{args.dataset}.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"[timesfm] saved control metrics: {metrics_path}")
    summary_df = _summarize_metrics(metrics_df)
    summary_path = metrics_dir / f"timesfm_control_summary_{args.dataset}.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"[timesfm] saved control summary (best/median): {summary_path}")


if __name__ == "__main__":
    main()
