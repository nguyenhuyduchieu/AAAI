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
import pandas as pd


def _timesfm_gaps() -> tuple[float, float]:
    vals_s, vals_v = [], []
    for ds in ["electricity", "weather", "ett", "traffic"]:
        df = pd.read_csv(Path("outputs/metrics") / f"timesfm_control_metrics_{ds}.csv")
        def get(weight: str, metric: str) -> float:
            s = df[
                (df["weight_mode"] == weight)
                & (df["label_mode"] == "na")
                & (df["metric"] == metric)
                & (df["perturb_setting"] == "clean")
            ]["value"]
            return float(s.mean())
        vals_s.append(100.0 * (get("pretrained", "CSS_seasonality_bal_acc") - get("random-reset", "CSS_seasonality_bal_acc")))
        vals_v.append(100.0 * (get("pretrained", "CSS_volatility_bal_acc") - get("random-reset", "CSS_volatility_bal_acc")))
    return float(np.mean(vals_s)), float(np.mean(vals_v))


def main() -> None:
    out = Path("outputs/figures/paper/fig_css_gap_vs_params.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    gap_season_tsfm, gap_vol_tsfm = _timesfm_gaps()

    models = ["TinyTimeMixer", "Chronos-Small", "MOMENT-Large", "TimesFM 2.0", "Moirai"]
    params_m = [5, 46, 385, 500, 311]
    gap_season = [None, None, None, gap_season_tsfm, None]
    gap_vol = [None, None, None, gap_vol_tsfm, None]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.log10(params_m)
    for i, (m, gs, gv) in enumerate(zip(models, gap_season, gap_vol)):
        if gs is not None:
            ax.scatter(x[i], gs, marker="o", s=110, c=C.blue, zorder=5, edgecolors="black", linewidths=0.6, label="Season" if i == 3 else "")
            ax.scatter(x[i], gv, marker="s", s=110, c=C.gold, zorder=5, edgecolors="black", linewidths=0.6, label="Volatility" if i == 3 else "")
            ax.annotate(m, (x[i], gs + 0.4), fontsize=10)
        else:
            ax.scatter(x[i], 0, marker="x", s=80, color=C.gray, zorder=5)
            ax.annotate(f"{m}\n(planned)", (x[i], 0.6), fontsize=9, color=C.gray_dark, ha="center")
    ax.set_xlabel("log10(Parameters)")
    ax.set_ylabel("CSS gap (Pre - Rand, pp)")
    ax.axhline(0, color=C.gray, linestyle=":", lw=0.9)
    ax.set_title("CSS gap vs. model scale")
    ax.legend()
    fig.tight_layout()
    paper_style.save_paper_figure(fig, out)
    plt.close(fig)


if __name__ == "__main__":
    main()
