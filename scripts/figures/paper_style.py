"""Shared Matplotlib style for paper figures: readable type, sharp PNG, classic colors.

Use before importing pyplot:

    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import paper_style
    paper_style.apply_paper_style()

    import matplotlib.pyplot as plt
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import matplotlib as mpl

# High-DPI PNG alongside PDF
PAPER_RASTER_DPI = 300

# Matplotlib tab10–style palette (classic publication defaults)
C = SimpleNamespace(
    blue="#1f77b4",
    blue_dark="#0b5394",
    orange="#ff7f0e",
    red="#d62728",
    green="#2ca02c",
    green_dark="#217a2a",
    purple="#9467bd",
    teal="#17becf",
    gold="#bcbd22",
    gray="#7f7f7f",
    gray_dark="#444444",
    ink="#333333",
)


def apply_paper_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "figure.facecolor": "white",
            "savefig.dpi": PAPER_RASTER_DPI,
            "savefig.transparent": False,
            "savefig.facecolor": "white",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.labelcolor": "black",
            "axes.titlecolor": "black",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "xtick.color": "black",
            "ytick.color": "black",
            "text.color": "black",
            "legend.fontsize": 10,
            "figure.titlesize": 13,
            "axes.linewidth": 1.0,
            "lines.linewidth": 1.65,
            "patch.linewidth": 1.0,
            "grid.linewidth": 0.8,
            "grid.color": "#b0b0b0",
            "grid.alpha": 0.35,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_paper_figure(fig, pdf_path: str | Path, *, png: bool = True, dpi: int | None = None) -> None:
    """Write vector PDF and optional high-DPI PNG next to it."""
    pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    raster = dpi if dpi is not None else PAPER_RASTER_DPI
    fc = mpl.rcParams["figure.facecolor"]
    fig.savefig(pdf_path, bbox_inches="tight", facecolor=fc)
    if png:
        fig.savefig(
            pdf_path.with_suffix(".png"),
            bbox_inches="tight",
            dpi=raster,
            format="png",
            facecolor=fc,
        )
