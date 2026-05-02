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
import pandas as pd


def _ltc(dataset: str, weight: str) -> float:
    path = Path("outputs/metrics") / f"timesfm_control_metrics_{dataset}.csv"
    df = pd.read_csv(path)
    s = df[
        (df["weight_mode"] == weight)
        & (df["label_mode"] == "na")
        & (df["metric"] == "LTC_step")
        & (df["perturb_setting"] == "clean")
    ]["value"]
    return float(s.mean())


def main() -> None:
    out = Path("outputs/figures/paper/fig_umap_pre_vs_rand.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    base = Path("outputs/figures")
    pre_img = base / "figure_timesfm_umap_electricity_pretrained.png"
    rnd_img = base / "figure_timesfm_umap_electricity_random-reset.png"
    if not pre_img.exists() or not rnd_img.exists():
        raise FileNotFoundError("Missing required UMAP images for pre vs rand.")

    ltc_pre = _ltc("electricity", "pretrained")
    ltc_rand = _ltc("electricity", "random-reset")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5))
    ax1.imshow(mpimg.imread(pre_img), interpolation="bilinear")
    ax1.set_title(f"Pretrained  LTC_step={ltc_pre:.2f}")
    ax1.axis("off")
    ax2.imshow(mpimg.imread(rnd_img), interpolation="bilinear")
    ax2.set_title(f"Random      LTC_step={ltc_rand:.2f}")
    ax2.axis("off")
    fig.suptitle("Shared-view UMAP comparison (Electricity)")
    fig.tight_layout()
    paper_style.save_paper_figure(fig, out)
    plt.close(fig)


if __name__ == "__main__":
    main()
