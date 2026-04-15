from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ExperimentConfig
from src.data.datasets import load_dataset
from src.eval.metrics import compute_gsp, compute_lcs, compute_tsi
from src.eval.perturbation import apply_masking, apply_noise, apply_time_warp
from src.latents.visualize import (
    project_umap,
    project_umap_3d,
    save_baseline_performance,
    save_latent_scatter3d_html,
    save_perturbation_bar,
    save_scatter,
)
from src.models.baseline_config import BaselineTrainConfig
from src.models.baselines import resolve_torch_device, run_baselines_walk_forward
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TSFM latent geometry benchmark")
    parser.add_argument("--dataset", type=str, default="electricity")
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--context-length", type=int, default=96)
    parser.add_argument("--splits", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lstm-hidden", type=int, default=128)
    parser.add_argument("--lstm-layers", type=int, default=2)
    parser.add_argument("--tcn-channels", type=int, default=128)
    parser.add_argument("--tcn-blocks", type=int, default=3)
    parser.add_argument("--transformer-d-model", type=int, default=128)
    parser.add_argument("--transformer-nhead", type=int, default=8)
    parser.add_argument("--transformer-layers", type=int, default=3)
    parser.add_argument("--transformer-dropout", type=float, default=0.1)
    parser.add_argument(
        "--no-scheduler",
        action="store_true",
        help="Disable OneCycleLR (use constant lr schedule from AdamW only).",
    )
    parser.add_argument("--max-samples", type=int, default=300000)
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers (0 = main process only).")
    parser.add_argument(
        "--no-pin-memory",
        action="store_true",
        help="Disable pin_memory (CUDA only; ignored on MPS).",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable mixed precision (CUDA/MPS autocast).",
    )
    parser.add_argument(
        "--compile-torch",
        action="store_true",
        help="Try torch.compile on CUDA (experimental speedup).",
    )
    parser.add_argument("--log-every", type=int, default=5, help="Print training loss every N epochs.")
    parser.add_argument(
        "--html-3d",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export interactive Plotly 3D UMAP/PCA HTML (standalone).",
    )
    parser.add_argument("--html-3d-max-points", type=int, default=12000, help="Max markers in HTML (evenly spaced in time).")
    parser.add_argument(
        "--html-3d-trajectory-points",
        type=int,
        default=4000,
        help="Max points along temporal path polyline in HTML.",
    )
    return parser.parse_args()


def _to_series(frame: pd.DataFrame, target_col: str) -> np.ndarray:
    return frame[target_col].astype(float).to_numpy()


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig(dataset=args.dataset, horizon=args.horizon, context_length=args.context_length)
    set_seed(cfg.random_seed)
    device = resolve_torch_device()
    print(f"[pipeline] dataset={cfg.dataset} horizon={cfg.horizon} context={cfg.context_length}")
    print(f"[pipeline] torch device={device}")

    train_cfg = BaselineTrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        tcn_channels=args.tcn_channels,
        tcn_blocks=args.tcn_blocks,
        transformer_d_model=args.transformer_d_model,
        transformer_nhead=args.transformer_nhead,
        transformer_layers=args.transformer_layers,
        transformer_dropout=args.transformer_dropout,
        use_scheduler=not args.no_scheduler,
        num_workers=args.num_workers,
        pin_memory=not args.no_pin_memory,
        use_amp=not args.no_amp,
        compile_torch=args.compile_torch,
        log_every=args.log_every,
    )

    print("[pipeline] loading dataset...")
    ds = load_dataset(cfg.dataset)
    series = _to_series(ds.frame, ds.target_col)
    print(f"[pipeline] loaded dataset rows={len(ds.frame)} target={ds.target_col}")

    print("[pipeline] running baseline walk-forward...")
    results, latent_bank = run_baselines_walk_forward(
        series,
        context_length=cfg.context_length,
        horizon=cfg.horizon,
        n_splits=args.splits,
        train_cfg=train_cfg,
        max_samples=args.max_samples,
        verbose=True,
    )

    metrics_df = pd.DataFrame([r.__dict__ for r in results])
    metrics_csv = Path("outputs/metrics") / f"baseline_metrics_{cfg.dataset}.csv"
    metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"[pipeline] saved baseline metrics: {metrics_csv}")

    paper_rows: list[dict] = []
    model_order = ["transformer", "lstm", "tcn"]

    for model_name in model_order:
        clean_emb = latent_bank[model_name]
        if len(clean_emb) == 0:
            continue
        time_idx = np.arange(len(clean_emb), dtype=np.float64)
        phase_labels = (np.arange(len(clean_emb)) % 8).astype(int)

        fig1_coords = project_umap(clean_emb, random_state=cfg.random_seed)
        fig1_path = Path("outputs/figures") / f"figure1_latent_umap_{cfg.dataset}_{model_name}.png"
        save_scatter(
            fig1_coords,
            time_idx,
            fig1_path,
            title=f"Figure 1 - Latent ({model_name}), color=time",
        )
        print(f"[pipeline] saved figure1: {fig1_path}")

        if args.html_3d:
            coords_3d = project_umap_3d(clean_emb, random_state=cfg.random_seed)
            html_path = Path("outputs/figures") / f"figure1_latent_3d_{cfg.dataset}_{model_name}.html"
            save_latent_scatter3d_html(
                coords_3d,
                time_idx,
                html_path,
                title=f"Latent 3D — {cfg.dataset} / {model_name}",
                subtitle="Markers evenly subsampled in time; thick line = chronological trajectory (geometry of dynamics)",
                max_points=args.html_3d_max_points,
                trajectory_max_points=args.html_3d_trajectory_points,
                include_plotlyjs=True,
            )
            print(f"[pipeline] saved interactive 3D HTML: {html_path}")

        noisy = apply_noise(clean_emb[:, None, :]).squeeze(1)
        masked = apply_masking(clean_emb[:, None, :]).squeeze(1)
        warped = apply_time_warp(clean_emb[:, None, :]).squeeze(1)

        lcs_clean = compute_lcs(clean_emb, phase_labels)
        lcs_noisy = compute_lcs(noisy, phase_labels)
        lcs_masked = compute_lcs(masked, phase_labels)
        lcs_warped = compute_lcs(warped, phase_labels)

        tsi_noisy = compute_tsi(clean_emb, noisy)
        tsi_masked = compute_tsi(clean_emb, masked)
        tsi_warped = compute_tsi(clean_emb, warped)

        gsp_noisy = compute_gsp(lcs_clean, lcs_noisy)
        gsp_masked = compute_gsp(lcs_clean, lcs_masked)
        gsp_warped = compute_gsp(lcs_clean, lcs_warped)

        fig2_path = Path("outputs/figures") / f"figure2_perturbation_{cfg.dataset}_{model_name}.png"
        save_perturbation_bar(
            {
                "LCS_clean": lcs_clean,
                "LCS_noise": lcs_noisy,
                "LCS_mask": lcs_masked,
                "LCS_warp": lcs_warped,
            },
            fig2_path,
        )
        print(f"[pipeline] saved figure2: {fig2_path}")

        for metric_name, value in [
            ("LCS_clean", lcs_clean),
            ("LCS_noise", lcs_noisy),
            ("LCS_mask", lcs_masked),
            ("LCS_warp", lcs_warped),
            ("TSI_noise", tsi_noisy),
            ("TSI_mask", tsi_masked),
            ("TSI_warp", tsi_warped),
            ("GSP_noise", gsp_noisy),
            ("GSP_mask", gsp_masked),
            ("GSP_warp", gsp_warped),
        ]:
            paper_rows.append({"model": model_name, "metric": metric_name, "value": value})

    fig3_path = Path("outputs/figures") / f"figure3_baselines_{cfg.dataset}.png"
    save_baseline_performance(metrics_df, fig3_path)
    print(f"[pipeline] saved figure3: {fig3_path}")

    summary_csv = Path("outputs/metrics") / f"paper_metrics_{cfg.dataset}.csv"
    summary = pd.DataFrame(paper_rows)
    summary.to_csv(summary_csv, index=False)
    print(f"[pipeline] saved paper metrics: {summary_csv}")


if __name__ == "__main__":
    main()
