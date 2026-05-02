from __future__ import annotations

from pathlib import Path
import sys

_d = Path(__file__).resolve().parent
if str(_d) not in sys.path:
    sys.path.insert(0, str(_d))
import paper_style

paper_style.apply_paper_style()

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def main() -> None:
    base = Path("outputs/figures")
    out_dir = base / "paper"
    out_dir.mkdir(parents=True, exist_ok=True)

    items = [
        ("Electricity", base / "figure_timesfm_umap_electricity_pretrained.png"),
        ("ETTh1", base / "figure_timesfm_umap_ett_pretrained.png"),
        ("Traffic", base / "figure_timesfm_umap_traffic_pretrained.png"),
        ("Weather", base / "figure_timesfm_umap_weather_pretrained.png"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.5))
    for ax, (title, path) in zip(axes.flat, items):
        if not path.exists():
            raise FileNotFoundError(f"Missing panel image: {path}")
        ax.imshow(mpimg.imread(path), interpolation="bilinear")
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    fig.suptitle("UMAP projections across datasets (TimesFM pretrained)", fontsize=13)
    fig.tight_layout()
    paper_style.save_paper_figure(fig, out_dir / "fig_umap_alldatasets.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()

