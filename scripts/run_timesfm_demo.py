from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import TimesFmModelForPrediction

from src.data.datasets import load_dataset
from src.latents.visualize import project_umap, project_umap_3d, save_latent_scatter3d_html, save_scatter
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TimesFM latent extraction demo")
    parser.add_argument("--dataset", type=str, default="electricity")
    parser.add_argument("--model-id", type=str, default="google/timesfm-2.0-500m-pytorch")
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--max-windows", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_windows(series: np.ndarray, context_length: int, max_windows: int) -> np.ndarray:
    n = len(series)
    if n <= context_length:
        raise ValueError("Series too short for requested context length")
    starts = np.linspace(0, n - context_length - 1, num=max_windows, dtype=int)
    x = np.stack([series[s : s + context_length] for s in starts], axis=0).astype(np.float32)
    return x


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device()

    print(f"[timesfm] loading dataset={args.dataset}")
    ds = load_dataset(args.dataset)
    series = ds.frame[ds.target_col].astype(float).to_numpy()

    windows = build_windows(series, context_length=args.context_length, max_windows=args.max_windows)
    # Fold-specific normalization for stable inference.
    mu = windows.mean(axis=1, keepdims=True)
    sigma = windows.std(axis=1, keepdims=True) + 1e-6
    windows = (windows - mu) / sigma

    print(f"[timesfm] model={args.model_id} device={device} windows={len(windows)}")
    model = TimesFmModelForPrediction.from_pretrained(args.model_id)
    model = model.to(device)
    model.eval()

    past_values = torch.from_numpy(windows).to(device)
    freq = torch.zeros((past_values.shape[0],), dtype=torch.long, device=device)

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
    embeddings = last_hidden.mean(axis=1)  # [B, D]
    time_idx = np.arange(len(embeddings), dtype=float)

    out_dir = Path("outputs/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    coords2 = project_umap(embeddings, random_state=args.seed)
    png_path = out_dir / f"figure_timesfm_umap_{args.dataset}.png"
    save_scatter(coords2, time_idx, png_path, title=f"TimesFM latent 2D ({args.dataset})")

    coords3 = project_umap_3d(embeddings, random_state=args.seed)
    html_path = out_dir / f"figure_timesfm_3d_{args.dataset}.html"
    save_latent_scatter3d_html(
        coords3,
        time_idx,
        html_path,
        title=f"TimesFM latent 3D ({args.dataset})",
        subtitle="Hidden-state sequence mean per window",
        max_points=2000,
        trajectory_max_points=600,
        include_plotlyjs=True,
    )

    print(f"[timesfm] saved 2D plot: {png_path}")
    print(f"[timesfm] saved 3D html: {html_path}")


if __name__ == "__main__":
    main()
