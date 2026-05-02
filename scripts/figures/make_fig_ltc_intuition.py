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


def _make_path(rng: np.random.Generator, n: int, drift: tuple[float, float], noise: float) -> np.ndarray:
    steps = rng.normal(loc=np.array(drift), scale=noise, size=(n, 2))
    return np.cumsum(steps, axis=0)


def main() -> None:
    out = Path("outputs/figures/paper/fig_ltc_intuition.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    n = 14
    rand_path = _make_path(rng, n, drift=(0.0, 0.0), noise=1.0)
    pre_path = _make_path(rng, n, drift=(0.6, 0.3), noise=0.5)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    items = [
        (ax1, rand_path, "Random-weight\nLTC_step low, LTC_dir ~ 0", C.orange),
        (ax2, pre_path, "Pretrained\nLTC_step high, LTC_dir > 0", C.blue),
    ]
    for ax, path, title, color in items:
        ax.scatter(path[:, 0], path[:, 1], color=color, zorder=5, s=48, alpha=0.9, edgecolors="black", linewidths=0.45)
        for i in range(len(path) - 1):
            ax.annotate(
                "",
                xy=path[i + 1],
                xytext=path[i],
                arrowprops=dict(arrowstyle="->", color=color, lw=1.6, mutation_scale=13),
            )
        for i, (x, y) in enumerate(path):
            ax.text(x + 0.05, y + 0.05, str(i + 1), fontsize=9.5, color=C.gray_dark)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Embedding dim 1")
        ax.set_ylabel("Embedding dim 2")
        ax.axis("equal")

    fig.tight_layout()
    paper_style.save_paper_figure(fig, out)
    plt.close(fig)


if __name__ == "__main__":
    main()
