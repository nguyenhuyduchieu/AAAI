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
from sklearn.linear_model import LogisticRegression


def _make_concept(rng: np.random.Generator, n: int, sep: float) -> tuple[np.ndarray, np.ndarray]:
    y = rng.integers(0, 2, size=n)
    x1 = rng.normal(loc=(y * sep), scale=1.0, size=n)
    x2 = rng.normal(loc=((1 - y) * sep * 0.2), scale=1.0, size=n)
    return np.column_stack([x1, x2]), y


def main() -> None:
    out = Path("outputs/figures/paper/fig_css_probe.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(12)

    concepts = [
        ("Seasonality", 2.0, 78.6, True),
        ("Volatility", 1.5, 70.9, True),
        ("Trend", 0.4, 67.9, False),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6))
    for ax, (name, sep, css, sig) in zip(axes, concepts):
        p2, y = _make_concept(rng, 600, sep)
        clf = LogisticRegression(C=1.0, max_iter=1000).fit(p2, y)
        xx, yy = np.meshgrid(
            np.linspace(p2[:, 0].min() - 0.5, p2[:, 0].max() + 0.5, 180),
            np.linspace(p2[:, 1].min() - 0.5, p2[:, 1].max() + 0.5, 180),
        )
        z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)
        ax.contourf(xx, yy, z, levels=[0, 0.5, 1], colors=[C.purple, C.gold], alpha=0.22)
        ax.contour(xx, yy, z, levels=[0.5], colors=C.gray, linestyles="--", linewidths=1.1)
        for k, color in enumerate([C.purple, C.gold]):
            mask = y == k
            ax.scatter(p2[mask, 0], p2[mask, 1], color=color, s=10, alpha=0.5, edgecolors="none")
        sig_text = "✓" if sig else "†(n.s.)"
        ax.set_title(f"{name}\nCSS = {css:.1f}%  {sig_text}", fontsize=12)
        ax.axis("off")

    fig.tight_layout()
    paper_style.save_paper_figure(fig, out)
    plt.close(fig)


if __name__ == "__main__":
    main()
