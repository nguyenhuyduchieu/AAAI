#!/usr/bin/env python3
"""Sweep RNG seeds through run_timesfm_demo and summarize PDS/FSS Spearman raw_corr (Pre vs Rand).

Writes one metrics folder per seed so runs do not overwrite each other. After a sweep (or with
--aggregate-only), prints a full table plus summary statistics.

The row with the largest (Pre - Rand) gap is reported as an exploratory variance bound — not as
a recommended seed for publications without reporting the full sweep or pre-specified protocol.

Example (expensive: full TimesFM forward per seed):

    PYTHONPATH=. python scripts/geodesic_corr_seed_sweep.py \\
        --dataset electricity --seed-start 0 --seed-end 4 --fast-dev

Aggregate existing outputs only:

    PYTHONPATH=. python scripts/geodesic_corr_seed_sweep.py \\
        --dataset electricity --aggregate-only \\
        --sweep-root outputs/geodesic_seed_sweep
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=str, default="electricity")
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--seed-end", type=int, default=4, help="Inclusive end seed.")
    p.add_argument("--k", type=int, default=20, help="PDS perturb_setting clean_k{k}.")
    p.add_argument(
        "--sweep-root",
        type=str,
        default="outputs/geodesic_seed_sweep",
        help="Root directory; each run uses {sweep-root}/{dataset}/seed_{n}/",
    )
    p.add_argument("--fast-dev", action="store_true", help="Pass --fast-dev to run_timesfm_demo.")
    p.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Do not run the demo; only scan existing seed_* folders under sweep-root/dataset.",
    )
    p.add_argument("--extra-demo-args", type=str, default="", help="Extra args passed to run_timesfm_demo.py.")
    return p.parse_args()


def _pick(df: pd.DataFrame, weight: str, metric: str, perturb: str) -> float:
    s = df[
        (df["weight_mode"] == weight)
        & (df["label_mode"] == "na")
        & (df["metric"] == metric)
        & (df["perturb_setting"] == perturb)
    ]["value"]
    if len(s) == 0:
        raise ValueError(f"Missing rows: {weight=} {metric=} {perturb=}")
    return float(s.mean())


def extract_row(csv_path: Path, k: int) -> dict[str, float]:
    df = pd.read_csv(csv_path)
    pds_m = "PDS_spearman_mae_havg_raw_corr"
    fss_m = "FSS_spearman_havg_raw_corr"
    pert_pds = f"clean_k{k}"
    return {
        "pre_pds": _pick(df, "pretrained", pds_m, pert_pds),
        "rnd_pds": _pick(df, "random-reset", pds_m, pert_pds),
        "pre_fss": _pick(df, "pretrained", fss_m, "clean"),
        "rnd_fss": _pick(df, "random-reset", fss_m, "clean"),
    }


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    demo = repo_root / "scripts" / "run_timesfm_demo.py"
    dataset = args.dataset
    base = Path(args.sweep_root) / dataset

    if not args.aggregate_only:
        base.mkdir(parents=True, exist_ok=True)
        extra = args.extra_demo_args.split() if args.extra_demo_args.strip() else []
        for seed in range(args.seed_start, args.seed_end + 1):
            metrics_dir = base / f"seed_{seed}"
            metrics_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                str(demo),
                "--dataset",
                dataset,
                "--seed",
                str(seed),
                "--weight-mode",
                "both",
                "--metrics-dir",
                str(metrics_dir),
            ]
            if args.fast_dev:
                cmd.append("--fast-dev")
            cmd.extend(extra)
            print("[sweep] running:", " ".join(cmd), flush=True)
            subprocess.run(cmd, cwd=str(repo_root), check=True)

    rows: list[dict[str, float | int]] = []
    def _seed_key(p: Path) -> int:
        parts = p.name.split("_")
        return int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else -1

    for sub in sorted(base.glob("seed_*"), key=_seed_key):
        csv_path = sub / f"timesfm_control_metrics_{dataset}.csv"
        if not csv_path.is_file():
            print(f"[sweep] skip (no csv): {csv_path}", file=sys.stderr)
            continue
        seed = int(sub.name.split("_")[1])
        r = extract_row(csv_path, args.k)
        gap_pds = r["pre_pds"] - r["rnd_pds"]
        gap_fss = r["pre_fss"] - r["rnd_fss"]
        rows.append(
            {
                "seed": seed,
                **r,
                "gap_pds": gap_pds,
                "gap_fss": gap_fss,
                "gap_sum": gap_pds + gap_fss,
            }
        )

    if not rows:
        print("No rows to summarize.", file=sys.stderr)
        sys.exit(1)

    table = pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    print("\n--- summary (exploratory) ---")
    for col in ["gap_pds", "gap_fss", "gap_sum"]:
        print(f"{col}: mean={table[col].mean():.5f} std={table[col].std():.5f} min={table[col].min():.5f} max={table[col].max():.5f}")

    i_pds = int(table["gap_pds"].idxmax())
    i_fss = int(table["gap_fss"].idxmax())
    i_sum = int(table["gap_sum"].idxmax())
    print("\nLargest gap_pds at seed:", int(table.loc[i_pds, "seed"]), "gap_pds=", float(table.loc[i_pds, "gap_pds"]))
    print("Largest gap_fss at seed:", int(table.loc[i_fss, "seed"]), "gap_fss=", float(table.loc[i_fss, "gap_fss"]))
    print("Largest gap_sum at seed:", int(table.loc[i_sum, "seed"]), "gap_sum=", float(table.loc[i_sum, "gap_sum"]))

    out_csv = base / f"geodesic_corr_seed_summary_k{args.k}.csv"
    table.to_csv(out_csv, index=False)
    print(f"\nWrote: {out_csv}")


if __name__ == "__main__":
    main()
