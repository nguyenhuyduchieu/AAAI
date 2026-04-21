from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export compact camera-ready geometry table.")
    parser.add_argument("--dataset", type=str, default="electricity")
    parser.add_argument("--metrics-file", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics_file = (
        Path(args.metrics_file)
        if args.metrics_file
        else Path(f"outputs/metrics/timesfm_geometry_summary_{args.dataset}.csv")
    )
    out_file = Path(f"outputs/metrics/timesfm_geometry_table_{args.dataset}.csv")
    summary = pd.read_csv(metrics_file)

    wanted_prefixes = ("FSS_", "PDS_", "LTC_", "LLS_", "MCS_", "CSS_", "TCS_")
    table = summary[
        summary["metric"].str.startswith(wanted_prefixes) & summary["metric"].str.contains("_score|LTC_|LLS_|MCS_|CSS_|TCS_")
    ].copy()
    table = table.sort_values(["label_mode", "metric", "weight_mode"]).reset_index(drop=True)
    table["delta_pretrained_minus_random"] = table.groupby(["label_mode", "metric"])["median"].transform(
        lambda s: float(s.iloc[0] - s.iloc[-1]) if len(s) >= 2 else float("nan")
    )
    table.to_csv(out_file, index=False)
    print(f"[geometry-table] saved: {out_file}")


if __name__ == "__main__":
    main()

