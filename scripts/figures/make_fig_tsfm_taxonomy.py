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


def _node(ax, x: float, y: float, text: str, color: str, w: float = 2.4, h: float = 0.8) -> None:
    rect = FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02", edgecolor=color, facecolor=color, alpha=0.18, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x, y, text, ha="center", va="center", fontsize=10)


def _edge(ax, x1: float, y1: float, x2: float, y2: float) -> None:
    ax.plot([x1, x2], [y1, y2], color=C.gray, linewidth=1.0, alpha=0.75)


def main() -> None:
    out = Path("outputs/figures/paper/fig_tsfm_taxonomy.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13.5, 5.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")

    _node(ax, 6, 7.2, "TSFMs", C.gray, w=2.0, h=0.9)

    fams = [
        ("Masked Enc-Dec", C.blue, 2.2),
        ("Decoder-only AR", C.teal, 4.8),
        ("Language-inspired", C.orange, 7.4),
        ("Efficient / MoE", C.purple, 10.0),
    ]
    for name, c, x in fams:
        _node(ax, x, 5.6, name, c)
        _edge(ax, 6, 6.75, x, 6.0)

    models = {
        2.2: ["MOMENT", "Moirai", "Moirai-MoE"],
        4.8: ["TimesFM", "Timer"],
        7.4: ["Chronos", "Lag-Llama", "LLMTime"],
        10.0: ["TinyTimeMixer", "Time-MoE"],
    }
    for x, ms in models.items():
        y0 = 4.0
        for i, m in enumerate(ms):
            y = y0 - i * 1.1
            _node(ax, x, y, m, C.gray, w=2.0, h=0.65)
            _edge(ax, x, 5.2, x, y + 0.35)

    ax.set_title("Taxonomy of TSFM architectural families", fontsize=13, pad=8)
    fig.tight_layout()
    paper_style.save_paper_figure(fig, out)
    plt.close(fig)


if __name__ == "__main__":
    main()
