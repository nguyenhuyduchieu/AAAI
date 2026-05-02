from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _pick(df: pd.DataFrame, weight: str, metric: str, perturb: str = "clean") -> float:
    s = df[
        (df["weight_mode"] == weight)
        & (df["label_mode"] == "na")
        & (df["metric"] == metric)
        & (df["perturb_setting"] == perturb)
    ]["value"]
    if len(s) == 0:
        raise RuntimeError(f"Missing {weight=} {metric=} {perturb=}")
    return float(s.mean())


def main() -> None:
    metrics_dir = Path("outputs/metrics")
    out_dir = Path("outputs/figures/paper")
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = ["electricity", "weather", "ett", "traffic"]
    ds_labels = ["Electricity", "Weather", "ETTh1", "Traffic"]
    concepts = ["trend", "seasonality", "volatility"]
    concept_labels = ["Trend", "Seasonality", "Volatility"]

    pre_data = np.zeros((3, 4), dtype=float)
    rnd_data = np.zeros((3, 4), dtype=float)
    pre_sig = np.zeros((3, 4), dtype=bool)
    rnd_sig = np.zeros((3, 4), dtype=bool)

    for j, ds in enumerate(datasets):
        df = pd.read_csv(metrics_dir / f"timesfm_control_metrics_{ds}.csv")
        for i, c in enumerate(concepts):
            m_acc = f"CSS_{c}_bal_acc"
            m_pv = f"CSS_{c}_pvalue"
            pre_data[i, j] = 100.0 * _pick(df, "pretrained", m_acc)
            rnd_data[i, j] = 100.0 * _pick(df, "random-reset", m_acc)
            pre_sig[i, j] = _pick(df, "pretrained", m_pv) < 0.05
            rnd_sig[i, j] = _pick(df, "random-reset", m_pv) < 0.05

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 3.8), sharey=True)

    for ax, data, sig, title in [
        (ax1, pre_data, pre_sig, "Pretrained"),
        (ax2, rnd_data, rnd_sig, "Random"),
    ]:
        sns.heatmap(
            data,
            ax=ax,
            annot=False,
            cmap="RdYlGn",
            vmin=40,
            vmax=90,
            center=50,
            xticklabels=ds_labels,
            yticklabels=concept_labels,
            cbar=(ax is ax2),
        )
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                txt = f"{data[i, j]:.1f}"
                if not sig[i, j]:
                    txt += "†"
                ax.text(j + 0.5, i + 0.5, txt, ha="center", va="center", fontsize=8.5, fontweight="bold" if sig[i, j] else "normal")
        ax.set_title(title)

    fig.tight_layout()
    fig.savefig(out_dir / "fig_probing_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

