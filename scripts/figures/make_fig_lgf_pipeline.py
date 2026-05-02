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
        alpha=0.24,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h * 0.68, label, ha="center", va="center", fontsize=12, fontweight="bold", color=color)
    ax.text(x + w / 2, y + h * 0.28, sublabel, ha="center", va="center", fontsize=10, color=C.ink)


def _arrow(ax, x1: float, x2: float, y: float = 2.0) -> None:
    ax.annotate("", xy=(x2, y), xytext=(x1, y), arrowprops=dict(arrowstyle="->", color=C.gray, lw=1.5, alpha=0.85))


def main() -> None:
    out = Path("outputs/figures/paper/fig_lgf_pipeline.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(15, 4.6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.axis("off")

    ax.text(0.35, 2.0, "Input\nwindow $w_i$", ha="center", va="center", fontsize=11)
    _arrow(ax, 0.75, 1.2)

    _box(ax, 1.2, 1.2, 1.8, 1.6, "Encoder $f_\\theta$", "(frozen inference)", C.gray)
    _arrow(ax, 3.0, 3.5)

    ax.text(3.75, 2.0, "Embeddings\n$Z\\subset\\mathbb{R}^D$", ha="center", va="center", fontsize=11)
    _arrow(ax, 4.2, 4.8)

    stages = [
        (4.8, "Stage 1\nLTC", "LTC_step\nLTC_dir", C.blue),
        (6.8, "Stage 2\nRegularity", "LCS  TSI  GSP", C.teal),
        (8.8, "Stage 3\nReliability", "PDS   FSS", C.orange),
        (10.8, "Stage 4\nEncoding", "CSS(trend)\nCSS(season)\nCSS(vol)", C.purple),
    ]
    for x, lbl, sub, c in stages:
        _box(ax, x, 0.8, 1.8, 2.4, lbl, sub, c)
        if x < 10.8:
            _arrow(ax, x + 1.8, x + 2.0)

    ax.text(13.45, 2.0, "8 geometric\nmetrics", ha="center", va="center", fontsize=11, style="italic")

    fig.tight_layout()
    paper_style.save_paper_figure(fig, out)
    plt.close(fig)


if __name__ == "__main__":
    main()
