from __future__ import annotations

from pathlib import Path

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

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (name, sep, css, sig) in zip(axes, concepts):
        p2, y = _make_concept(rng, 600, sep)
        clf = LogisticRegression(C=1.0, max_iter=1000).fit(p2, y)
        xx, yy = np.meshgrid(
            np.linspace(p2[:, 0].min() - 0.5, p2[:, 0].max() + 0.5, 180),
            np.linspace(p2[:, 1].min() - 0.5, p2[:, 1].max() + 0.5, 180),
        )
        z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)
        ax.contourf(xx, yy, z, levels=[0, 0.5, 1], colors=["#8e44ad", "#e67e22"], alpha=0.12)
        ax.contour(xx, yy, z, levels=[0.5], colors="black", linestyles="--", linewidths=1.2)
        for k, color in enumerate(["#8e44ad", "#e67e22"]):
            mask = y == k
            ax.scatter(p2[mask, 0], p2[mask, 1], color=color, s=8, alpha=0.5)
        sig_text = "✓" if sig else "†(n.s.)"
        ax.set_title(f"{name}\nCSS = {css:.1f}%  {sig_text}", fontsize=9)
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
