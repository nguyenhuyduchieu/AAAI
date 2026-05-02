from __future__ import annotations

from pathlib import Path

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

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7))
    for ax, (title, path) in zip(axes, items):
        if not path.exists():
            raise FileNotFoundError(path)
        ax.imshow(mpimg.imread(path))
        ax.set_title(title, fontsize=9.5)
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_dir / "fig_stock_ood.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

