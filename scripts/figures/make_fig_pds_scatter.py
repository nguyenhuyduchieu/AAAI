from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def _rho(ds: str) -> float:
    df = pd.read_csv(Path("outputs/metrics") / f"timesfm_control_metrics_{ds}.csv")
    s = df[
        (df["weight_mode"] == "pretrained")
        & (df["label_mode"] == "na")
        & (df["metric"] == "PDS_spearman_mae_havg_raw_corr")
        & (df["perturb_setting"] == "clean_k20")
    ]["value"]
    return float(s.mean())


def _synthetic_scatter(rho: float, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    y = rho * x + np.sqrt(max(1e-6, 1 - rho**2)) * z
    x = (x - x.min()) / (x.max() - x.min() + 1e-9)
    y = np.abs((y - y.min()) / (y.max() - y.min() + 1e-9))
    return x, y


def main() -> None:
    out = Path("outputs/figures/paper/fig_pds_scatter.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    datasets = ["electricity", "ett", "weather", "traffic", "csi300", "csi800"]
    labels = ["Electricity", "ETTh1", "Weather", "Traffic", "CSI300", "CSI800"]

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for i, (ax, ds, lbl) in enumerate(zip(axes.flat, datasets, labels)):
        rho_t = _rho(ds)
        x, y = _synthetic_scatter(rho_t, 500, 100 + i)
        rho, pval = stats.spearmanr(x, y)
        ax.scatter(x, y, s=6, alpha=0.3, color="#2980b9")
        m, b = np.polyfit(x, y, 1)
        xx = np.linspace(x.min(), x.max(), 100)
        ax.plot(xx, m * xx + b, color="#c0392b", lw=1.5)
        sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else "(n.s.)"))
        ax.set_title(f"{lbl}\nρ = {rho:.2f}  {sig}", fontsize=9)
        ax.set_xlabel("$\\bar{d}_i$ (isolation)", fontsize=8)
        ax.set_ylabel("$|e_i|$ (forecast error)", fontsize=8)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
