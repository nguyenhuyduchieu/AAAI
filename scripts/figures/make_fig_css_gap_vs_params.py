from __future__ import annotations

from pathlib import Path

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

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.log10(params_m)
    for i, (m, gs, gv) in enumerate(zip(models, gap_season, gap_vol)):
        if gs is not None:
            ax.scatter(x[i], gs, marker="o", s=100, color="#2980b9", zorder=5, label="Season" if i == 3 else "")
            ax.scatter(x[i], gv, marker="s", s=100, color="#e67e22", zorder=5, label="Volatility" if i == 3 else "")
            ax.annotate(m, (x[i], gs + 0.4), fontsize=8)
        else:
            ax.scatter(x[i], 0, marker="x", s=80, color="grey", zorder=5)
            ax.annotate(f"{m}\n(planned)", (x[i], 0.6), fontsize=7, color="grey", ha="center")
    ax.set_xlabel("log10(Parameters)")
    ax.set_ylabel("CSS gap (Pre - Rand, pp)")
    ax.axhline(0, color="grey", linestyle=":", lw=0.9)
    ax.set_title("CSS gap vs. model scale")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
