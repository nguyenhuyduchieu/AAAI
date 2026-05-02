from __future__ import annotations

from pathlib import Path
import re
import sys

_d = Path(__file__).resolve().parent
if str(_d) not in sys.path:
    sys.path.insert(0, str(_d))
import paper_style
from paper_style import C

paper_style.apply_paper_style()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    out = Path("outputs/figures/paper/fig_ltc_by_layer.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    path = Path("outputs/metrics/timesfm_geometry_missing_validations_cross_dataset.csv")
    df = pd.read_csv(path)
    sub = df[(df["model"] == "tsfm_pretrained") & (df["metric"].str.contains(r"^CSS_layer\d+_seasonality_bal_acc$"))].copy()
    if sub.empty:
        raise RuntimeError("No layer-wise metrics found.")
    sub["layer"] = sub["metric"].map(lambda x: int(re.search(r"layer(\d+)", x).group(1)))
    pre = sub.groupby("layer")["value"].mean().sort_index()

    # Proxy random baseline from non-layer seasonality accuracy in control metrics.
    rand_vals = []
    for ds in ["electricity", "weather", "ett", "traffic"]:
        cdf = pd.read_csv(Path("outputs/metrics") / f"timesfm_control_metrics_{ds}.csv")
        s = cdf[
            (cdf["weight_mode"] == "random-reset")
            & (cdf["label_mode"] == "na")
            & (cdf["metric"] == "CSS_seasonality_bal_acc")
            & (cdf["perturb_setting"] == "clean")
        ]["value"]
        rand_vals.append(float(s.mean()))
    rand_mean = float(np.mean(rand_vals))
    rand = pd.Series([rand_mean] * len(pre), index=pre.index)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(pre.index, pre.values, "o-", label="Pretrained (layer-wise CSS proxy)", color=C.blue, lw=2.2, markersize=7, markeredgecolor="black", markeredgewidth=0.4)
    ax.plot(rand.index, rand.values, "s--", label="Random (global baseline)", color=C.orange, lw=2.0, markersize=6, markeredgecolor="black", markeredgewidth=0.35)
    ax.fill_between(pre.index, rand.values, pre.values, alpha=0.22, color=C.blue)
    ax.set_xlabel("Transformer layer index")
    ax.set_ylabel("Score")
    ax.set_title("Layer-wise representation dynamics (CSS proxy)")
    ax.legend()
    fig.tight_layout()
    paper_style.save_paper_figure(fig, out)
    plt.close(fig)


if __name__ == "__main__":
    main()
