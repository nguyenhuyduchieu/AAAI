from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def project_umap(embeddings: np.ndarray, random_state: int = 42) -> np.ndarray:
    """Project embeddings to 2D; prefer UMAP, fallback to PCA for stability."""
    try:
        import umap  # type: ignore

        reducer = umap.UMAP(n_components=2, random_state=random_state, n_neighbors=15, min_dist=0.1)
        return reducer.fit_transform(embeddings)
    except Exception:
        from sklearn.decomposition import PCA

        return PCA(n_components=2, random_state=random_state).fit_transform(embeddings)


def project_umap_3d(embeddings: np.ndarray, random_state: int = 42) -> np.ndarray:
    """Project embeddings to 3D; prefer UMAP, fallback to PCA."""
    try:
        import umap  # type: ignore

        reducer = umap.UMAP(n_components=3, random_state=random_state, n_neighbors=15, min_dist=0.1)
        return reducer.fit_transform(embeddings)
    except Exception:
        from sklearn.decomposition import PCA

        return PCA(n_components=3, random_state=random_state).fit_transform(embeddings)


def save_scatter(coords: np.ndarray, labels: np.ndarray, out_file: Path, title: str = "Latent Geometry (UMAP)") -> None:
    """Readable 2D latent plot: time-colored points + trajectory + start/end anchors."""
    out_file.parent.mkdir(parents=True, exist_ok=True)

    n = len(coords)
    if n == 0:
        return

    labels = np.asarray(labels).reshape(-1)
    if len(labels) != n:
        labels = np.arange(n, dtype=float)

    # Keep the figure readable for huge point clouds.
    max_points = 12000
    if n > max_points:
        idx = np.linspace(0, n - 1, num=max_points, dtype=int)
        c = coords[idx]
        lab = labels[idx]
    else:
        c = coords
        lab = labels

    # Trajectory with evenly sampled points.
    traj_n = int(min(len(c), 1500))
    traj_idx = np.linspace(0, len(c) - 1, num=max(2, traj_n), dtype=int)

    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    sc = ax.scatter(c[:, 0], c[:, 1], c=lab, s=12, cmap="viridis", alpha=0.85, linewidths=0)
    ax.plot(c[traj_idx, 0], c[traj_idx, 1], color="black", alpha=0.28, linewidth=1.2, label="temporal path")

    # Explicit anchors for interpretability.
    ax.scatter(c[0, 0], c[0, 1], c="#22c55e", s=90, marker="o", edgecolors="black", linewidths=0.8, label="start")
    ax.scatter(c[-1, 0], c[-1, 1], c="#ef4444", s=90, marker="X", edgecolors="black", linewidths=0.8, label="end")

    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("time / sample index")

    ax.set_title(title)
    ax.set_xlabel("UMAP / PCA dim 1")
    ax.set_ylabel("UMAP / PCA dim 2")
    ax.grid(alpha=0.2, linestyle="--")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_file, dpi=220)
    plt.close(fig)


def save_perturbation_bar(values: dict[str, float], out_file: Path) -> None:
    """Clear perturbation chart with value annotations."""
    out_file.parent.mkdir(parents=True, exist_ok=True)
    keys = list(values.keys())
    vals = [float(values[k]) for k in keys]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    bars = ax.bar(keys, vals, color=["#2563eb", "#22c55e", "#f59e0b", "#ef4444"][: len(keys)])
    ax.set_ylabel("Score")
    ax.set_title("Perturbation Stability")
    ax.grid(axis="y", alpha=0.2, linestyle="--")

    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.4f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_file, dpi=220)
    plt.close(fig)


def save_latent_scatter3d_html(
    coords_3d: np.ndarray,
    time_index: np.ndarray,
    out_file: Path,
    title: str = "Latent geometry (3D)",
    subtitle: str = "",
    max_points: int = 12000,
    trajectory_max_points: int = 4000,
    include_plotlyjs: bool = True,
) -> None:
    import plotly.graph_objects as go

    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    n = len(coords_3d)
    if n == 0:
        return

    t = np.asarray(time_index, dtype=np.float64).reshape(-1)
    if len(t) != n:
        t = np.arange(n, dtype=np.float64)

    if n > max_points:
        idx = np.linspace(0, n - 1, num=max_points, dtype=int)
        c = coords_3d[idx]
        ts = t[idx]
    else:
        c = coords_3d
        ts = t

    traj_n = int(min(n, max(2, trajectory_max_points)))
    tidx = np.linspace(0, n - 1, num=traj_n, dtype=int)
    ct = coords_3d[tidx]
    tt = t[tidx]

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=ct[:, 0], y=ct[:, 1], z=ct[:, 2],
                mode="lines", name="temporal path",
                line=dict(color="#111827", width=5),
                hovertemplate="trajectory<extra></extra>",
            ),
            go.Scatter3d(
                x=c[:, 0], y=c[:, 1], z=c[:, 2],
                mode="markers", name="windows",
                marker=dict(size=3.5, color=ts, colorscale="Viridis", opacity=0.88, showscale=True,
                            colorbar=dict(title="time index", len=0.65)),
                hovertemplate="t=%{marker.color:.0f}<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>",
            ),
            go.Scatter3d(
                x=[c[0, 0]], y=[c[0, 1]], z=[c[0, 2]], mode="markers", name="start",
                marker=dict(size=7, color="#22c55e", symbol="circle"),
            ),
            go.Scatter3d(
                x=[c[-1, 0]], y=[c[-1, 1]], z=[c[-1, 2]], mode="markers", name="end",
                marker=dict(size=7, color="#ef4444", symbol="diamond"),
            ),
        ]
    )

    text = title if not subtitle else f"{title}<br><sup>{subtitle}</sup>"
    fig.update_layout(
        template="plotly_white",
        title=dict(text=text, x=0.5),
        margin=dict(l=0, r=0, t=70, b=0),
        legend=dict(y=0.98, x=0.01),
        scene=dict(
            xaxis_title="dim 1",
            yaxis_title="dim 2",
            zaxis_title="dim 3",
            aspectmode="data",
            camera=dict(eye=dict(x=1.45, y=1.45, z=1.2)),
        ),
    )
    fig.write_html(str(out_file), include_plotlyjs=include_plotlyjs, full_html=True)


def save_baseline_performance(metrics_df: pd.DataFrame, out_file: Path) -> None:
    """Readable grouped bars with metric labels."""
    out_file.parent.mkdir(parents=True, exist_ok=True)

    grouped = metrics_df.groupby("model_name")[["mae", "rmse"]].mean().reset_index()
    x = np.arange(len(grouped))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    b1 = ax.bar(x - width / 2, grouped["mae"], width, label="MAE", color="#2563eb")
    b2 = ax.bar(x + width / 2, grouped["rmse"], width, label="RMSE", color="#f59e0b")

    ax.set_xticks(x)
    ax.set_xticklabels(grouped["model_name"])
    ax.set_title("Baseline Walk-Forward Performance (lower is better)")
    ax.set_ylabel("Error")
    ax.grid(axis="y", alpha=0.2, linestyle="--")
    ax.legend()

    for bars in (b1, b2):
        for b in bars:
            v = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_file, dpi=220)
    plt.close(fig)
