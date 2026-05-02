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
        ("Pretrained UMAP", base / "figure_timesfm_umap_csi300_pretrained.png"),
        ("Random UMAP", base / "figure_timesfm_umap_csi300_random-reset.png"),
        ("Error-colored UMAP", base / "figure_pds_error_umap_csi300.png"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.5))
    for ax, (title, path) in zip(axes, items):
        if not path.exists():
            raise FileNotFoundError(path)
        ax.imshow(mpimg.imread(path), interpolation="bilinear")
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    fig.tight_layout()
    paper_style.save_paper_figure(fig, out_dir / "fig_stock_ood.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()

