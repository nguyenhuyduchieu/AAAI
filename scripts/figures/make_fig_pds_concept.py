from __future__ import annotations

from pathlib import Path
import sys

_d = Path(__file__).resolve().parent
if str(_d) not in sys.path:
    sys.path.insert(0, str(_d))
import paper_style
from paper_style import C

paper_style.apply_paper_style()

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    out = Path("outputs/figures/paper/fig_pds_concept.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    rng = np.random.default_rng(7)

    dense_xy = rng.normal(loc=[1.5, 1.5], scale=0.25, size=(40, 2))
    dense_err = rng.uniform(0.02, 0.08, 40)
    sparse_xy = np.array([[3.8, 3.0], [4.2, 2.5], [3.5, 3.6]])
    sparse_err = np.array([0.45, 0.52, 0.38])

    ax1.scatter(dense_xy[:, 0], dense_xy[:, 1], s=dense_err * 800, c=C.blue, alpha=0.55, edgecolors="none", label="Low error")
    ax1.scatter(sparse_xy[:, 0], sparse_xy[:, 1], s=sparse_err * 800, c=C.gold, alpha=0.75, edgecolors="none", label="High error")
    iso = sparse_xy[0]
    ax1.add_patch(plt.Circle(iso, 0.6, fill=False, linestyle="--", edgecolor=C.orange, lw=1.4, alpha=0.95))
    ax1.annotate("Large $\\bar{d}_i$", xy=iso + [0, 0.65], ha="center", fontsize=10, color=C.orange)
    ax1.text(1.5, 0.7, "Dense region\n(small $\\bar{d}_i$, low error)", ha="center", fontsize=10, color=C.blue_dark)
    ax1.set_title("PDS - local density")
    ax1.legend(fontsize=10)
    ax1.axis("equal")
    ax1.axis("off")

    zi = np.array([1.0, 1.0])
    zj = np.array([1.4, 1.2])
    zk = np.array([3.5, 2.8])
    for pt, lbl in [(zi, "$z_i$"), (zj, "$z_j$"), (zk, "$z_k$")]:
        ax2.scatter(*pt, s=80, zorder=5, color=C.ink, alpha=0.9, edgecolors="black", linewidths=0.6)
        ax2.text(pt[0] + 0.07, pt[1] + 0.07, lbl, fontsize=11)
    ax2.annotate("", xy=zj, xytext=zi, arrowprops=dict(arrowstyle="<->", color=C.green_dark, lw=2, alpha=0.9))
    mid_ij = (zi + zj) / 2
    ax2.text(mid_ij[0] - 0.15, mid_ij[1] + 0.1, "$\\|z_i-z_j\\|$ small\n$|e_i-e_j|$ small", fontsize=9.5, color=C.green_dark)
    ax2.annotate("", xy=zk, xytext=zi, arrowprops=dict(arrowstyle="<->", color=C.red, lw=2, alpha=0.9))
    mid_ik = (zi + zk) / 2
    ax2.text(mid_ik[0] + 0.1, mid_ik[1], "$\\|z_i-z_k\\|$ large\n$|e_i-e_k|$ large", fontsize=9.5, color=C.red)
    ax2.set_title("FSS - global proximity")
    ax2.set_xlim(0, 5)
    ax2.set_ylim(0, 4)
    ax2.axis("off")

    fig.tight_layout()
    paper_style.save_paper_figure(fig, out)
    plt.close(fig)


if __name__ == "__main__":
    main()
