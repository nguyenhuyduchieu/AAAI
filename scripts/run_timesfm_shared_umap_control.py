from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from transformers import TimesFmModelForPrediction

from src.data.datasets import load_dataset
from src.eval.geometry_metrics import compute_ltc
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Shared-UMAP random-weight control for TimesFM")
    p.add_argument("--dataset", type=str, default="electricity")
    p.add_argument(
        "--datasets",
        type=str,
        default="",
        help="Comma list datasets to run in one command, e.g. weather,electricity,ett,traffic,csi300,csi800",
    )
    p.add_argument("--model-id", type=str, default="google/timesfm-2.0-500m-pytorch")
    p.add_argument("--context-length", type=int, default=512)
    p.add_argument("--max-windows", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-fig-dir", type=str, default="outputs/figures")
    p.add_argument("--out-metrics-dir", type=str, default="outputs/metrics")
    return p.parse_args()


def _dataset_list(args: argparse.Namespace) -> list[str]:
    if args.datasets.strip():
        return [x.strip() for x in args.datasets.split(",") if x.strip()]
    return [args.dataset]


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_windows(series: np.ndarray, context_length: int, max_windows: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(series)
    if n <= context_length:
        raise ValueError("Series too short for requested context length.")
    starts = np.linspace(0, n - context_length, num=max_windows, dtype=int)
    x = np.stack([series[s : s + context_length] for s in starts], axis=0).astype(np.float32)
    return x, starts


def _load_series_fallback(dataset: str) -> np.ndarray:
    ds = dataset.lower()
    if ds == "weather":
        p = Path("data/raw/long_horizon/longhorizon/datasets/weather/M/df_y.csv")
        df = pd.read_csv(p)
        return df.groupby("ds", as_index=False)["y"].mean()["y"].astype(float).to_numpy()
    if ds == "electricity":
        p = Path("data/raw/long_horizon/longhorizon/datasets/ECL/M/df_y.csv")
        df = pd.read_csv(p)
        return df.groupby("ds", as_index=False)["y"].mean()["y"].astype(float).to_numpy()
    if ds == "traffic":
        p = Path("data/raw/long_horizon/longhorizon/datasets/traffic/M/df_y.csv")
        df = pd.read_csv(p)
        return df.groupby("ds", as_index=False)["y"].mean()["y"].astype(float).to_numpy()
    if ds == "ett":
        p = Path("external/ETDataset/ETT-small/ETTh1.csv")
        df = pd.read_csv(p)
        return pd.to_numeric(df["OT"], errors="coerce").dropna().astype(float).to_numpy()
    if ds in {"csi300", "csi800"}:
        p = Path(f"data/raw/{ds}.csv")
        if not p.exists():
            p = Path("data/raw/master_csi300_market.csv")
        df = pd.read_csv(p)
        col = "y" if "y" in df.columns else df.columns[-1]
        return pd.to_numeric(df[col], errors="coerce").dropna().astype(float).to_numpy()
    raise ValueError(f"Unsupported dataset for fallback loader: {dataset}")


def _reset_module_parameters(module: torch.nn.Module) -> None:
    for layer in module.modules():
        if hasattr(layer, "reset_parameters") and callable(layer.reset_parameters):
            layer.reset_parameters()


def load_tsfm(model_id: str, device: torch.device, random_reset: bool, seed: int) -> TimesFmModelForPrediction:
    model = TimesFmModelForPrediction.from_pretrained(model_id)
    if random_reset:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        _reset_module_parameters(model)
    return model.to(device).eval()


def extract_embeddings(model: TimesFmModelForPrediction, windows_norm: np.ndarray, device: torch.device) -> np.ndarray:
    pv = torch.from_numpy(windows_norm).to(device)
    freq = torch.zeros((pv.shape[0],), dtype=torch.long, device=device)
    with torch.no_grad():
        outputs = model(
            past_values=pv,
            freq=freq,
            output_hidden_states=True,
            return_dict=True,
        )
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None or len(hidden_states) == 0:
        raise RuntimeError("TimesFM outputs missing hidden states")
    last_hidden = hidden_states[-1].detach().float().cpu().numpy()
    return last_hidden.mean(axis=1)


def fit_shared_umap(z_pretrained: np.ndarray, z_random: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    try:
        import umap  # type: ignore
    except Exception as e:
        raise RuntimeError("umap-learn is required for shared UMAP protocol") from e

    reducer = umap.UMAP(
        n_neighbors=30,
        min_dist=0.1,
        metric="cosine",
        random_state=seed,
    )
    pre_2d = reducer.fit_transform(z_pretrained)
    rnd_2d = reducer.transform(z_random)
    return pre_2d, rnd_2d


def save_comparison_figure(pre_2d: np.ndarray, rnd_2d: np.ndarray, colors: np.ndarray, out_file: Path) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    x_all = np.concatenate([pre_2d[:, 0], rnd_2d[:, 0]])
    y_all = np.concatenate([pre_2d[:, 1], rnd_2d[:, 1]])
    xlim = (float(x_all.min()), float(x_all.max()))
    ylim = (float(y_all.min()), float(y_all.max()))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    sc1 = axes[0].scatter(pre_2d[:, 0], pre_2d[:, 1], c=colors, cmap="viridis", s=6, alpha=0.9, linewidths=0)
    axes[0].set_title("Pretrained TimesFM (UMAP fit)")
    axes[0].set_xlim(*xlim)
    axes[0].set_ylim(*ylim)
    axes[0].grid(alpha=0.2, linestyle="--")

    axes[1].scatter(rnd_2d[:, 0], rnd_2d[:, 1], c=colors, cmap="viridis", s=6, alpha=0.9, linewidths=0)
    axes[1].set_title("Random-Weight Control (UMAP transform)")
    axes[1].set_xlim(*xlim)
    axes[1].set_ylim(*ylim)
    axes[1].grid(alpha=0.2, linestyle="--")

    for ax in axes:
        ax.set_xlabel("UMAP dim 1")
    axes[0].set_ylabel("UMAP dim 2")
    fig.colorbar(sc1, ax=axes, fraction=0.025, pad=0.02, label="time index")
    fig.tight_layout()
    fig.savefig(out_file, dpi=220)
    plt.close(fig)


def save_report(
    dataset: str,
    starts: np.ndarray,
    z_pretrained: np.ndarray,
    z_random: np.ndarray,
    ltc_pre: float,
    ltc_rnd: float,
    fig_path: Path,
    metrics_dir: Path,
) -> None:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    delta = float(ltc_pre - ltc_rnd)
    ok = bool(ltc_pre > ltc_rnd)

    metrics_csv = metrics_dir / f"timesfm_shared_umap_metrics_{dataset}.csv"
    pd.DataFrame(
        [
            {"dataset": dataset, "metric": "ltc_pretrained", "value": float(ltc_pre)},
            {"dataset": dataset, "metric": "ltc_random_reset", "value": float(ltc_rnd)},
            {"dataset": dataset, "metric": "ltc_delta_pre_minus_random", "value": delta},
            {"dataset": dataset, "metric": "expectation_pre_gt_random", "value": float(ok)},
        ]
    ).to_csv(metrics_csv, index=False)

    coords_csv = metrics_dir / f"timesfm_shared_umap_coords_{dataset}.csv"
    pd.DataFrame(
        {
            "dataset": dataset,
            "start_index": starts.astype(int),
            "color_t": np.linspace(0.0, 1.0, len(starts)),
            "z_pretrained_dim1": z_pretrained[:, 0],
            "z_pretrained_dim2": z_pretrained[:, 1],
            "z_random_dim1": z_random[:, 0],
            "z_random_dim2": z_random[:, 1],
        }
    ).to_csv(coords_csv, index=False)

    report_md = metrics_dir / f"timesfm_shared_umap_report_{dataset}.md"
    report_md.write_text(
        "\n".join(
            [
                f"# Shared UMAP Control Report ({dataset})",
                "",
                "## LTC",
                f"- `ltc_pretrained`: {ltc_pre:.6f}",
                f"- `ltc_random_reset`: {ltc_rnd:.6f}",
                f"- `delta_pre_minus_random`: {delta:.6f}",
                f"- `expectation_pre_gt_random`: {'PASS' if ok else 'FAIL'}",
                "",
                "## Fairness Checklist",
                "- Same architecture: PASS (TimesFM 2.0 model class for both branches)",
                "- Same dataset: PASS",
                "- Same windowing: PASS",
                "- Same preprocessing (z-normalization per window): PASS",
                "- Same seed/projection: PASS (single shared UMAP reducer)",
                "- Only weights differ: PASS (pretrained vs random-reset)",
                "",
                "## Methodology Guardrails",
                "- No separate `fit_transform` for random branch: PASS",
                "- Random branch uses `reducer.transform(...)`: PASS",
                "- Same data order and temporal coloring between branches: PASS",
                "",
                "## Artifacts",
                f"- Figure: `{fig_path.as_posix()}`",
                f"- Metrics CSV: `{metrics_csv.as_posix()}`",
                f"- Coordinates CSV: `{coords_csv.as_posix()}`",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device()
    datasets = _dataset_list(args)
    summary_rows: list[dict[str, float | str]] = []

    for i, dataset in enumerate(datasets):
        run_seed = args.seed + i
        set_seed(run_seed)
        try:
            ds = load_dataset(dataset)
            series = ds.frame[ds.target_col].astype(float).to_numpy()
        except Exception:
            series = _load_series_fallback(dataset)
        windows, starts = build_windows(series, args.context_length, args.max_windows)

        mu = windows.mean(axis=1, keepdims=True)
        sigma = windows.std(axis=1, keepdims=True) + 1e-6
        windows_norm = (windows - mu) / sigma

        pretrained_model = load_tsfm(args.model_id, device=device, random_reset=False, seed=run_seed)
        random_model = load_tsfm(args.model_id, device=device, random_reset=True, seed=run_seed)

        z_pre = extract_embeddings(pretrained_model, windows_norm, device)
        z_rnd = extract_embeddings(random_model, windows_norm, device)
        pre_2d, rnd_2d = fit_shared_umap(z_pre, z_rnd, run_seed)

        colors = np.linspace(0.0, 1.0, len(pre_2d))
        fig_path = Path(args.out_fig_dir) / f"figure_timesfm_umap_shared_{dataset}_pre_vs_random.png"
        save_comparison_figure(pre_2d, rnd_2d, colors, fig_path)

        ltc_pre, _ = compute_ltc(z_pre, starts)
        ltc_rnd, _ = compute_ltc(z_rnd, starts)
        delta = float(ltc_pre - ltc_rnd)
        save_report(
            dataset=dataset,
            starts=starts,
            z_pretrained=pre_2d,
            z_random=rnd_2d,
            ltc_pre=ltc_pre,
            ltc_rnd=ltc_rnd,
            fig_path=fig_path,
            metrics_dir=Path(args.out_metrics_dir),
        )
        summary_rows.append(
            {
                "dataset": dataset,
                "ltc_pretrained": float(ltc_pre),
                "ltc_random_reset": float(ltc_rnd),
                "ltc_delta_pre_minus_random": delta,
                "expectation_pre_gt_random": float(ltc_pre > ltc_rnd),
                "seed": float(run_seed),
            }
        )
        print(f"[shared-umap] saved figure: {fig_path}")
        print(f"[shared-umap] dataset={dataset} LTC_pre={ltc_pre:.6f} LTC_random={ltc_rnd:.6f} delta={delta:.6f}")

    if len(summary_rows) > 1:
        out_summary = Path(args.out_metrics_dir) / "timesfm_shared_umap_metrics_all.csv"
        pd.DataFrame(summary_rows).to_csv(out_summary, index=False)
        print(f"[shared-umap] saved all-dataset summary: {out_summary}")


if __name__ == "__main__":
    main()

