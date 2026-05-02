from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    out = Path("outputs/figures/paper/fig_pds_concept.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    rng = np.random.default_rng(7)

    dense_xy = rng.normal(loc=[1.5, 1.5], scale=0.25, size=(40, 2))
    dense_err = rng.uniform(0.02, 0.08, 40)
    sparse_xy = np.array([[3.8, 3.0], [4.2, 2.5], [3.5, 3.6]])
    sparse_err = np.array([0.45, 0.52, 0.38])

    ax1.scatter(dense_xy[:, 0], dense_xy[:, 1], s=dense_err * 800, c="#3498db", alpha=0.6, label="Low error")
    ax1.scatter(sparse_xy[:, 0], sparse_xy[:, 1], s=sparse_err * 800, c="#e67e22", alpha=0.8, label="High error")
    iso = sparse_xy[0]
    ax1.add_patch(plt.Circle(iso, 0.6, fill=False, linestyle="--", edgecolor="#e67e22", lw=1.5))
    ax1.annotate("Large $\\bar{d}_i$", xy=iso + [0, 0.65], ha="center", fontsize=8, color="#e67e22")
    ax1.text(1.5, 0.7, "Dense region\n(small $\\bar{d}_i$, low error)", ha="center", fontsize=8, color="#2980b9")
    ax1.set_title("PDS - local density")
    ax1.legend(fontsize=8)
    ax1.axis("equal")
    ax1.axis("off")

    zi = np.array([1.0, 1.0])
    zj = np.array([1.4, 1.2])
    zk = np.array([3.5, 2.8])
    for pt, lbl in [(zi, "$z_i$"), (zj, "$z_j$"), (zk, "$z_k$")]:
        ax2.scatter(*pt, s=80, zorder=5, color="#2c3e50")
        ax2.text(pt[0] + 0.07, pt[1] + 0.07, lbl, fontsize=10)
    ax2.annotate("", xy=zj, xytext=zi, arrowprops=dict(arrowstyle="<->", color="#27ae60", lw=2))
    mid_ij = (zi + zj) / 2
    ax2.text(mid_ij[0] - 0.15, mid_ij[1] + 0.1, "$\\|z_i-z_j\\|$ small\n$|e_i-e_j|$ small", fontsize=7.5, color="#27ae60")
    ax2.annotate("", xy=zk, xytext=zi, arrowprops=dict(arrowstyle="<->", color="#c0392b", lw=2))
    mid_ik = (zi + zk) / 2
    ax2.text(mid_ik[0] + 0.1, mid_ik[1], "$\\|z_i-z_k\\|$ large\n$|e_i-e_k|$ large", fontsize=7.5, color="#c0392b")
    ax2.set_title("FSS - global proximity")
    ax2.set_xlim(0, 5)
    ax2.set_ylim(0, 4)
    ax2.axis("off")

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
