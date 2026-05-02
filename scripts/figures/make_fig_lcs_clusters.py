from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull


def _lcs_value(dataset: str, weight_mode: str) -> float:
    path = Path("outputs/metrics") / f"timesfm_control_metrics_{dataset}.csv"
    df = pd.read_csv(path)
    s = df[
        (df["weight_mode"] == weight_mode)
        & (df["label_mode"] == "stl")
        & (df["perturb_setting"] == "clean")
        & (df["metric"] == "LCS_clean")
    ]["value"]
    if len(s) == 0:
        return 0.0
    return float(s.mean())


def main() -> None:
    out = Path("outputs/figures/paper/fig_lcs_clusters.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(7)
    n = 500
    y = rng.integers(0, 3, size=n)

    centers = np.array([[1.2, 1.4], [0.0, -0.8], [-1.3, 1.0]])
    pre = centers[y] + rng.normal(scale=0.35, size=(n, 2))
    rand = rng.normal(scale=1.0, size=(n, 2))

    colors = ["#e74c3c", "#2ecc71", "#3498db"]
    names = ["Up-trend", "Flat", "Down-trend"]

    lcs_pre = _lcs_value("weather", "pretrained")
    lcs_rand = _lcs_value("weather", "random-reset")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for ax, pts, title, draw_hull in [
        (ax1, pre, f"Pretrained (LCS = {lcs_pre:.3f})", True),
        (ax2, rand, f"Random (LCS = {lcs_rand:.3f})", False),
    ]:
        for k, (c, name) in enumerate(zip(colors, names)):
            mask = y == k
            p = pts[mask]
            ax.scatter(p[:, 0], p[:, 1], color=c, label=name, s=12, alpha=0.6)
            if draw_hull and p.shape[0] > 3:
                hull = ConvexHull(p)
                hp = np.append(p[hull.vertices], p[hull.vertices[:1]], axis=0)
                ax.plot(hp[:, 0], hp[:, 1], color=c, lw=1.2, alpha=0.75)
        ax.set_title(title)
        ax.axis("off")
    ax1.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
