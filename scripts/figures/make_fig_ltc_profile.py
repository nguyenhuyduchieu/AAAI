from __future__ import annotations

import argparse
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=str, default="geometry_extensions")
    return p.parse_args()


def _ltc_step_from_file(path: Path) -> dict[str, float]:
    df = pd.read_csv(path)
    sub = df[df["metric"] == "LTC_step_baseline_cmp"][["model", "value"]]
    return {str(r.model): float(r.value) for r in sub.itertuples(index=False)}


def main() -> None:
    _ = parse_args()
    metrics_dir = Path("outputs/metrics")
    out_dir = Path("outputs/figures/paper")
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = ["electricity", "weather", "ett", "traffic"]
    pretty = ["Electricity", "Weather", "ETTh1", "Traffic"]

    values = {m: [] for m in ["tsfm_pretrained", "lstm", "tcn", "transformer"]}
    for ds in datasets:
        path = metrics_dir / f"timesfm_geometry_missing_validations_{ds}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        rows = _ltc_step_from_file(path)
        for m in values:
            values[m].append(rows[m])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.8))

    x = np.arange(len(pretty))
    w = 0.22
    ax1.bar(x - 1.5 * w, values["tsfm_pretrained"], w, label="TimesFM (pre)", color=C.blue)
    ax1.bar(x - 0.5 * w, values["lstm"], w, label="LSTM", color=C.gray)
    ax1.bar(x + 0.5 * w, values["tcn"], w, label="TCN", color=C.gray_dark)
    ax1.bar(x + 1.5 * w, values["transformer"], w, label="Transformer", color=C.gold)
    ax1.set_xticks(x)
    ax1.set_xticklabels(pretty)
    ax1.set_ylabel("LTC_step")
    ax1.set_title("Across datasets")
    ax1.legend(fontsize=10)

    elec_vals = [
        values["tsfm_pretrained"][0],
        values["lstm"][0],
        values["tcn"][0],
        values["transformer"][0],
    ]
    names = ["TimesFM (pre)", "LSTM", "TCN", "Transformer"]
    colors = [C.blue, C.gray, C.gray_dark, C.gold]
    ax2.bar(names, elec_vals, color=colors)
    ax2.set_title("Electricity (all models)")
    ax2.set_ylabel("LTC_step")
    ax2.tick_params(axis="x", labelrotation=20)

    fig.tight_layout()
    paper_style.save_paper_figure(fig, out_dir / "fig_ltc_profile.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()

