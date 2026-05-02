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
    p.add_argument("--k", type=int, default=20)
    return p.parse_args()


def _pick(df: pd.DataFrame, weight: str, metric: str, perturb: str) -> float:
    s = df[
        (df["weight_mode"] == weight)
        & (df["label_mode"] == "na")
        & (df["metric"] == metric)
        & (df["perturb_setting"] == perturb)
    ]["value"]
    if len(s) == 0:
        raise RuntimeError(f"Missing {weight=} {metric=} {perturb=}")
    return float(s.mean())


def _ci_raw_from_score(df: pd.DataFrame, weight: str, base_metric: str, perturb: str) -> tuple[float, float]:
    # score = -raw_corr in pipeline => raw_ci_low = -score_ci_high, raw_ci_high = -score_ci_low
    low = _pick(df, weight, f"{base_metric}_ci_low", perturb)
    high = _pick(df, weight, f"{base_metric}_ci_high", perturb)
    return -high, -low


def main() -> None:
    args = parse_args()
    out_dir = Path("outputs/figures/paper")
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = Path("outputs/metrics")

    datasets = ["electricity", "ett", "weather", "traffic"]
    labels = ["Electricity", "ETTh1", "Weather", "Traffic"]

    pre_pds, pre_fss, rnd_pds, rnd_fss = [], [], [], []
    pre_pds_err, pre_fss_err, rnd_pds_err, rnd_fss_err = [], [], [], []

    for ds in datasets:
        df = pd.read_csv(metrics_dir / f"timesfm_control_metrics_{ds}.csv")

        pds_metric = "PDS_spearman_mae_havg"
        fss_metric = "FSS_spearman_havg"

        p_pre = _pick(df, "pretrained", f"{pds_metric}_raw_corr", f"clean_k{args.k}")
        p_rnd = _pick(df, "random-reset", f"{pds_metric}_raw_corr", f"clean_k{args.k}")
        f_pre = _pick(df, "pretrained", f"{fss_metric}_raw_corr", "clean")
        f_rnd = _pick(df, "random-reset", f"{fss_metric}_raw_corr", "clean")
        pre_pds.append(p_pre)
        rnd_pds.append(p_rnd)
        pre_fss.append(f_pre)
        rnd_fss.append(f_rnd)

        p_pre_ci = _ci_raw_from_score(df, "pretrained", pds_metric, f"clean_k{args.k}")
        p_rnd_ci = _ci_raw_from_score(df, "random-reset", pds_metric, f"clean_k{args.k}")
        f_pre_ci = _ci_raw_from_score(df, "pretrained", fss_metric, "clean")
        f_rnd_ci = _ci_raw_from_score(df, "random-reset", fss_metric, "clean")
        pre_pds_err.append([p_pre - p_pre_ci[0], p_pre_ci[1] - p_pre])
        rnd_pds_err.append([p_rnd - p_rnd_ci[0], p_rnd_ci[1] - p_rnd])
        pre_fss_err.append([f_pre - f_pre_ci[0], f_pre_ci[1] - f_pre])
        rnd_fss_err.append([f_rnd - f_rnd_ci[0], f_rnd_ci[1] - f_rnd])

    x = np.arange(len(labels))
    w = 0.2
    fig, ax = plt.subplots(figsize=(12, 4.8))

    ax.bar(
        x - 1.5 * w,
        pre_pds,
        w,
        label="Pre PDS",
        yerr=np.array(pre_pds_err).T,
        capsize=2,
        color=C.blue_dark,
    )
    ax.bar(
        x - 0.5 * w,
        pre_fss,
        w,
        label="Pre FSS",
        yerr=np.array(pre_fss_err).T,
        capsize=2,
        color=C.blue,
    )
    ax.bar(
        x + 0.5 * w,
        rnd_pds,
        w,
        label="Rand PDS",
        yerr=np.array(rnd_pds_err).T,
        capsize=2,
        color=C.red,
    )
    ax.bar(
        x + 1.5 * w,
        rnd_fss,
        w,
        label="Rand FSS",
        yerr=np.array(rnd_fss_err).T,
        capsize=2,
        color=C.orange,
    )
    ax.axhline(0.0, color=C.gray, linestyle="--", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Spearman raw correlation")
    ax.set_title("Geodesic reliability: PDS vs FSS")
    ax.legend(ncol=2, fontsize=10)

    fig.tight_layout()
    paper_style.save_paper_figure(fig, out_dir / "fig_geodesic_reliability.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()

