from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _grouped(ax, labels: list[str], pre: list[float], rnd: list[float], ylabel: str, add_zero: bool = False, add_chance: bool = False) -> None:
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w / 2, pre, w, color="#2980b9", label="Pre")
    ax.bar(x + w / 2, rnd, w, color="#e74c3c", label="Rand")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    if add_zero:
        ax.axhline(0, color="black", linestyle="--", lw=1.0)
    if add_chance:
        ax.axhline(50, color="grey", linestyle=":", lw=0.9)


def main() -> None:
    out = Path("outputs/figures/paper/fig_ood_summary.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 2, figsize=(10, 7), sharey="row")
    fig.subplots_adjust(hspace=0.5, wspace=0.35)

    ind = ["Elec.", "Weath.", "ETTh1", "Traffic"]
    ood = ["CSI300", "CSI800"]

    _grouped(axes[0, 0], ind, [4.64, 5.10, 5.02, 3.77], [3.90, 3.52, 3.70, 2.55], "LTC_step")
    _grouped(axes[0, 1], ood, [4.24, 4.30], [3.80, 3.82], "LTC_step")

    _grouped(axes[1, 0], ind, [82.6, 71.8, 77.6, 82.6], [64.3, 65.9, 72.8, 71.0], "CSS Season (%)", add_chance=True)
    _grouped(axes[1, 1], ood, [74.8, 70.0], [58.1, 55.9], "CSS Season (%)", add_chance=True)

    _grouped(axes[2, 0], ind, [0.41, 0.04, 0.11, 0.25], [-0.34, -0.27, 0.10, 0.08], "PDS  $\\rho_s$", add_zero=True)
    _grouped(axes[2, 1], ood, [0.02, -0.09], [-0.37, -0.38], "PDS  $\\rho_s$", add_zero=True)

    axes[0, 0].set_title("In-distribution", fontweight="bold")
    axes[0, 1].set_title("OOD (equity)", fontweight="bold", color="#c0392b")
    for r in range(3):
        for spine in axes[r, 1].spines.values():
            spine.set_edgecolor("#c0392b")
            spine.set_linewidth(1.5)

    axes[0, 0].legend(fontsize=8)
    fig.suptitle("OOD validation: LTC and CSS transfer; PDS collapses", fontsize=10, fontweight="bold")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
