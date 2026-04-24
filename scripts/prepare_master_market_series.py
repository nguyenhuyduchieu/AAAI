from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare univariate MASTER market series for existing metric pipeline.")
    p.add_argument("--master-csv", type=str, default="external/MASTER/data/csi_market_information.csv")
    p.add_argument("--universe", type=str, default="SH000300", choices=["SH000300", "SH000905", "SH000906"])
    p.add_argument("--feature-contains", type=str, default="$close/Ref($close,1)-1")
    p.add_argument("--output-name", type=str, default="master_csi300_market")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    src = Path(args.master_csv)
    if not src.exists():
        raise FileNotFoundError(f"MASTER csv not found: {src}")

    raw = pd.read_csv(src)
    if raw.empty:
        raise RuntimeError("MASTER market CSV is empty.")

    # Row 0 stores feature formulas; row 1 onward stores datetime + values.
    feature_row = raw.iloc[0]
    data = raw.iloc[1:].copy()
    if "Unnamed: 0" not in data.columns:
        raise RuntimeError("Expected datetime-like column 'Unnamed: 0' in MASTER market CSV.")

    candidate_cols: list[str] = []
    for c in data.columns:
        if c == "Unnamed: 0":
            continue
        expr = str(feature_row.get(c, ""))
        if args.universe in expr and args.feature_contains in expr:
            candidate_cols.append(c)
    if not candidate_cols:
        raise RuntimeError(
            f"No feature column matched universe={args.universe} and contains='{args.feature_contains}'."
        )
    target_col = candidate_cols[0]

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(data["Unnamed: 0"], errors="coerce"),
            "y": pd.to_numeric(data[target_col], errors="coerce"),
        }
    ).dropna()
    out["unique_id"] = args.output_name
    out = out[["unique_id", "date", "y"]]

    out_path = Path("data/raw") / f"{args.output_name}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[master-prepare] selected feature column: {target_col}")
    print(f"[master-prepare] rows: {len(out)}")
    print(f"[master-prepare] saved: {out_path}")


if __name__ == "__main__":
    main()

