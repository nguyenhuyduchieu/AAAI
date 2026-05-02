from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def _box(ax, x: float, y: float, w: float, h: float, label: str, sublabel: str, color: str) -> None:
    rect = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.1",
        linewidth=1.5,
        edgecolor=color,
        facecolor=color,
        alpha=0.12,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h * 0.68, label, ha="center", va="center", fontsize=10, fontweight="bold", color=color)
    ax.text(x + w / 2, y + h * 0.28, sublabel, ha="center", va="center", fontsize=7.5, color="#444444")


def _arrow(ax, x1: float, x2: float, y: float = 2.0) -> None:
    ax.annotate("", xy=(x2, y), xytext=(x1, y), arrowprops=dict(arrowstyle="->", color="black", lw=1.5))


def main() -> None:
    out = Path("outputs/figures/paper/fig_lgf_pipeline.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.axis("off")

    ax.text(0.35, 2.0, "Input\nwindow $w_i$", ha="center", va="center", fontsize=9)
    _arrow(ax, 0.75, 1.2)

    _box(ax, 1.2, 1.2, 1.8, 1.6, "Encoder $f_\\theta$", "(frozen inference)", "#555555")
    _arrow(ax, 3.0, 3.5)

    ax.text(3.75, 2.0, "Embeddings\n$Z\\subset\\mathbb{R}^D$", ha="center", va="center", fontsize=9)
    _arrow(ax, 4.2, 4.8)

    stages = [
        (4.8, "Stage 1\nLTC", "LTC_step\nLTC_dir", "#1f77b4"),
        (6.8, "Stage 2\nRegularity", "LCS  TSI  GSP", "#2ca02c"),
        (8.8, "Stage 3\nReliability", "PDS   FSS", "#ff7f0e"),
        (10.8, "Stage 4\nEncoding", "CSS(trend)\nCSS(season)\nCSS(vol)", "#9467bd"),
    ]
    for x, lbl, sub, c in stages:
        _box(ax, x, 0.8, 1.8, 2.4, lbl, sub, c)
        if x < 10.8:
            _arrow(ax, x + 1.8, x + 2.0)

    ax.text(13.45, 2.0, "8 geometric\nmetrics", ha="center", va="center", fontsize=9, style="italic")

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
