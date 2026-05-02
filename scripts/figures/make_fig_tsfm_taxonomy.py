from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def _node(ax, x: float, y: float, text: str, color: str, w: float = 2.4, h: float = 0.8) -> None:
    rect = FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02", edgecolor=color, facecolor=color, alpha=0.18, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x, y, text, ha="center", va="center", fontsize=8)


def _edge(ax, x1: float, y1: float, x2: float, y2: float) -> None:
    ax.plot([x1, x2], [y1, y2], color="#555555", linewidth=1.0)


def main() -> None:
    out = Path("outputs/figures/paper/fig_tsfm_taxonomy.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")

    _node(ax, 6, 7.2, "TSFMs", "#7f8c8d", w=2.0, h=0.9)

    fams = [
        ("Masked Enc-Dec", "#3498db", 2.2),
        ("Decoder-only AR", "#2ecc71", 4.8),
        ("Language-inspired", "#e67e22", 7.4),
        ("Efficient / MoE", "#9b59b6", 10.0),
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
            _node(ax, x, y, m, "#bdc3c7", w=2.0, h=0.65)
            _edge(ax, x, 5.2, x, y + 0.35)

    ax.set_title("Taxonomy of TSFM architectural families", fontsize=11, pad=6)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
