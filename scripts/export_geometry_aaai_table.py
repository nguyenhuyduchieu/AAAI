from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export compact AAAI LaTeX table for geometry metrics.")
    parser.add_argument("--datasets", type=str, default="electricity,weather,ett,traffic")
    parser.add_argument("--output-prefix", type=str, default="outputs/metrics/timesfm_geometry_aaai_compact")
    return parser.parse_args()


def _fmt_med_ci(median: float, low: float, high: float, digits: int = 3) -> str:
    return f"{median:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def _extract_score_row(summary: pd.DataFrame, dataset: str, weight_mode: str, metric: str) -> tuple[float, float, float] | None:
    row = summary[
        (summary["dataset"] == dataset)
        & (summary["weight_mode"] == weight_mode)
        & (summary["label_mode"] == "na")
        & (summary["metric"] == metric)
    ]
    if row.empty:
        return None
    r = row.iloc[0]
    return float(r["median"]), float(r["worst"]), float(r["best"])


def build_table_rows(summary: pd.DataFrame, datasets: list[str]) -> pd.DataFrame:
    records: list[dict[str, str]] = []
    horizons = ["96", "192", "336", "720", "avg"]
    for ds in datasets:
        global_metrics = {
            "LTC_step": "LTC_step",
            "LTC_dir": "LTC_dir",
            "LLS": "LLS_avgk",
            "MCS": "MCS_k20",
            "TCS": "TCS_avg",
            "CSS_trend": "CSS_trend_bal_acc",
            "CSS_season": "CSS_seasonality_bal_acc",
            "CSS_vol": "CSS_volatility_bal_acc",
        }

        global_pre: dict[str, str] = {}
        global_rnd: dict[str, str] = {}
        for alias, metric_name in global_metrics.items():
            t_pre = _extract_score_row(summary, ds, "pretrained", metric_name)
            t_rnd = _extract_score_row(summary, ds, "random-reset", metric_name)
            global_pre[alias] = "-" if t_pre is None else _fmt_med_ci(*t_pre)
            global_rnd[alias] = "-" if t_rnd is None else _fmt_med_ci(*t_rnd)

        for hz in horizons:
            metrics = {
                "FSS": f"FSS_spearman_h{hz}_score",
                "PDS": f"PDS_spearman_mae_h{hz}_score",
            }
            rec: dict[str, str] = {"Dataset": ds, "Horizon": hz}
            for model_name, weight_mode in [("Pretrained", "pretrained"), ("Random", "random-reset")]:
                for alias, metric_name in metrics.items():
                    triple = _extract_score_row(summary, ds, weight_mode, metric_name)
                    rec[f"{alias}_{model_name}"] = "-" if triple is None else _fmt_med_ci(*triple)
            for alias in global_metrics:
                rec[f"{alias}_Pretrained"] = global_pre[alias]
                rec[f"{alias}_Random"] = global_rnd[alias]
            records.append(rec)

    return pd.DataFrame(records)


def to_latex(df: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\caption{Cross-dataset latent-geometry metrics (median with [worst, best] across sweep settings).}")
    lines.append("\\label{tab:geometry_cross_dataset_compact}")
    lines.append("\\begin{tabular}{ll" + "c" * (len(df.columns) - 2) + "}")
    lines.append("\\toprule")
    cols = " & ".join(df.columns)
    lines.append(cols + " \\\\")
    lines.append("\\midrule")
    for _, row in df.iterrows():
        vals = [str(row[c]) for c in df.columns]
        lines.append(" & ".join(vals) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    summary_parts = []
    for ds in datasets:
        p = Path(f"outputs/metrics/timesfm_geometry_summary_{ds}.csv")
        if not p.exists():
            raise FileNotFoundError(f"Missing summary file: {p}")
        summary_parts.append(pd.read_csv(p))
    summary = pd.concat(summary_parts, ignore_index=True)
    compact = build_table_rows(summary, datasets=datasets)

    out_csv = Path(f"{args.output_prefix}.csv")
    out_tex = Path(f"{args.output_prefix}.tex")
    compact.to_csv(out_csv, index=False)
    out_tex.write_text(to_latex(compact), encoding="utf-8")
    print(f"[aaai-table] saved csv: {out_csv}")
    print(f"[aaai-table] saved tex: {out_tex}")


if __name__ == "__main__":
    main()

