from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_MODELS = [
    "LightGTS",
    "NeurIPS2023-One-Fits-All",
    "Sundial",
    "Time-MoE",
    "UniTime",
    "VisionTS",
    "moment",
    "tirex",
    "uni2ts",
    "chronos-forecasting",
]

DEFAULT_DATASETS = ["weather", "electricity", "ett", "traffic", "csi300", "csi800"]


@dataclass
class RunResult:
    model: str
    dataset: str
    success: bool
    elapsed_sec: float
    command: str
    env: str
    error: str
    extra: dict[str, float]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified external baseline runner")
    p.add_argument("--models", type=str, default=",".join(DEFAULT_MODELS))
    p.add_argument("--datasets", type=str, default=",".join(DEFAULT_DATASETS))
    p.add_argument("--horizon", type=int, default=96)
    p.add_argument("--timeout-sec", type=int, default=1800)
    p.add_argument("--env-root", type=str, default=".venv_baselines")
    p.add_argument("--out-dir", type=str, default="outputs/metrics")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _split_csv(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _python_for_model(model: str, env_root: Path) -> tuple[str, str]:
    m = model.lower()
    if m in {"sundial", "time-moe", "chronos-forecasting"}:
        env = "env_foundation"
    elif m in {"uni2ts"}:
        env = "env_uni2ts"
    elif m in {"moment"}:
        env = "env_moment"
    else:
        env = "env_legacy"
    return str(env_root / env / "bin" / "python"), env


def _model_command(model: str, dataset: str, horizon: int) -> list[str]:
    m = model.lower()
    if m == "sundial":
        in_npy = f"/tmp/sundial_{dataset}_in.npy"
        out_npy = f"/tmp/sundial_{dataset}_out.npy"
        return ["python", "scripts/sundial_infer.py", "--input", in_npy, "--output", out_npy, "--horizon", str(horizon)]
    if m == "time-moe":
        data = _prepare_time_moe_csv(dataset)
        return [
            "python",
            "external/baselines/Time-MoE/run_eval.py",
            "-d",
            data,
            "-p",
            str(horizon),
        ]
    if m == "chronos-forecasting":
        return [
            "python",
            "external/baselines/uni2ts/project/benchmarks/run_chronos.py",
            "--model_path",
            "amazon/chronos-t5-small",
            "--dataset",
            "nn5_daily_without_missing",
            "--save_dir",
            "outputs/metrics/tmp_chronos",
            "--run_name",
            dataset,
        ]
    if m == "uni2ts":
        return [
            "python",
            "external/baselines/uni2ts/cli/eval.py",
            "data=lsf_test",
            "model=moirai_1.0_R_small",
            "data.dataset_name=ETTh1",
            "data.prediction_length=96",
            f"run_name={dataset}",
        ]
    if m == "visionts":
        return ["python", "external/baselines/VisionTS/long_term_tsf/run.py", "--help"]
    if m == "lightgts":
        return ["python", "external/baselines/LightGTS/zero_shot.py", "--help"]
    if m == "neurips2023-one-fits-all":
        return ["python", "external/baselines/NeurIPS2023-One-Fits-All/Long-term_Forecasting/main.py", "--help"]
    if m == "unitime":
        return ["python", "external/baselines/UniTime/run.py", "--help"]
    if m == "moment":
        return [
            "python",
            "-c",
            (
                "import torch; "
                "from momentfm import MOMENTPipeline; "
                "MOMENTPipeline.from_pretrained('AutonLab/MOMENT-1-small', "
                "model_kwargs={'task_name':'forecasting','forecast_horizon':96}); "
                "print('moment_loaded')"
            ),
        ]
    if m == "tirex":
        return ["python", "external/baselines/tirex/benchmark/benchmark.py", "--help"]
    return ["python", "-c", f"print('unsupported model: {model}')"]


def _run_one(model: str, dataset: str, horizon: int, timeout_sec: int, env_root: Path, dry_run: bool) -> RunResult:
    py_bin, env = _python_for_model(model, env_root)
    parts = _model_command(model, dataset, horizon)
    command = [py_bin if p == "python" else p for p in parts]
    shell_cmd = " ".join(command)
    start = time.time()

    if dry_run:
        return RunResult(model, dataset, True, 0.0, shell_cmd, env, "", {})

    if model.lower() == "sundial":
        in_npy = Path(f"/tmp/sundial_{dataset}_in.npy")
        np.save(in_npy, np.random.randn(8, 96).astype(np.float32))
    env_vars = None
    if model.lower() in {"chronos-forecasting", "uni2ts"}:
        env_vars = dict(**__import__("os").environ)
        env_vars["PYTHONPATH"] = str(Path("external/baselines/uni2ts/src")) + ":" + env_vars.get("PYTHONPATH", "")
    if model.lower() == "visionts":
        env_vars = dict(**__import__("os").environ) if env_vars is None else env_vars
        env_vars["PYTHONPATH"] = str(Path("external/baselines/VisionTS")) + ":" + env_vars.get("PYTHONPATH", "")
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=timeout_sec, env=env_vars)
        elapsed = time.time() - start
        ok = proc.returncode == 0
        err = (proc.stderr or proc.stdout or "").strip()[-1000:]
        metrics: dict[str, float] = {}
        return RunResult(model, dataset, ok, elapsed, shell_cmd, env, err, metrics)
    except subprocess.TimeoutExpired:
        return RunResult(
            model=model,
            dataset=dataset,
            success=False,
            elapsed_sec=float(timeout_sec),
            command=shell_cmd,
            env=env,
            error=f"timeout_after_{timeout_sec}s",
            extra={},
        )


def _rows_from_results(results: Iterable[RunResult]) -> tuple[list[dict], list[dict]]:
    metrics_rows: list[dict] = []
    summary_rows: list[dict] = []
    for r in results:
        metrics_rows.extend(
            [
                {
                    "dataset": r.dataset,
                    "model": r.model,
                    "weight_mode": "na",
                    "label_mode": "na",
                    "stl_setting": "na",
                    "perturb_setting": "na",
                    "noise_scale": 0.0,
                    "mask_p": 0.0,
                    "warp_ratio": 0.0,
                    "metric": "success",
                    "value": 1.0 if r.success else 0.0,
                },
                {
                    "dataset": r.dataset,
                    "model": r.model,
                    "weight_mode": "na",
                    "label_mode": "na",
                    "stl_setting": "na",
                    "perturb_setting": "na",
                    "noise_scale": 0.0,
                    "mask_p": 0.0,
                    "warp_ratio": 0.0,
                    "metric": "elapsed_sec",
                    "value": round(r.elapsed_sec, 4),
                },
            ]
        )
        for k, v in r.extra.items():
            metrics_rows.append(
                {
                    "dataset": r.dataset,
                    "model": r.model,
                    "weight_mode": "na",
                    "label_mode": "na",
                    "stl_setting": "na",
                    "perturb_setting": "na",
                    "noise_scale": 0.0,
                    "mask_p": 0.0,
                    "warp_ratio": 0.0,
                    "metric": k,
                    "value": v,
                }
            )

        summary_rows.append(
            {
                "model": r.model,
                "dataset": r.dataset,
                "summary": "ok" if r.success else f"failed: {r.error}",
                "env": r.env,
                "command": r.command,
            }
        )
    return metrics_rows, summary_rows


def _summarize_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty:
        return pd.DataFrame(columns=["dataset", "model", "metric", "best", "median", "mean", "worst"])
    summary = (
        metrics_df.groupby(["dataset", "model", "metric"], as_index=False)["value"]
        .agg(best="max", median="median", mean="mean", worst="min")
    )
    return summary


def _prepare_time_moe_csv(dataset: str) -> str:
    if dataset == "ett":
        return "external/ETDataset/ETT-small/ETTh1.csv"
    out = Path("outputs/metrics/tmp_time_moe")
    out.mkdir(parents=True, exist_ok=True)
    source_map = {
        "weather": "data/raw/long_horizon/longhorizon/datasets/weather/M/df_y.csv",
        "electricity": "data/raw/long_horizon/longhorizon/datasets/ECL/M/df_y.csv",
        "traffic": "data/raw/long_horizon/longhorizon/datasets/traffic/M/df_y.csv",
    }
    if dataset in source_map:
        long_df = pd.read_csv(source_map[dataset])
        mapped = long_df.pivot(index="ds", columns="unique_id", values="y").reset_index().rename(columns={"ds": "date"})
    elif dataset in {"csi300", "csi800"}:
        src = f"data/raw/{dataset}.csv"
        if not Path(src).exists():
            src = "data/raw/master_csi300_market.csv"
        tmp = pd.read_csv(src)
        ts_col = "datetime" if "datetime" in tmp.columns else ("ds" if "ds" in tmp.columns else tmp.columns[0])
        val_col = "y" if "y" in tmp.columns else tmp.columns[-1]
        mapped = pd.DataFrame({"date": pd.to_datetime(tmp[ts_col]).astype(str), "value": pd.to_numeric(tmp[val_col], errors="coerce")})
    else:
        raise ValueError(f"Unsupported dataset for Time-MoE mapping: {dataset}")
    if len(mapped) > 5000:
        mapped = mapped.tail(5000).reset_index(drop=True)
    out_path = out / f"{dataset}.csv"
    mapped.to_csv(out_path, index=False)
    return str(out_path)


def main() -> None:
    args = parse_args()
    models = _split_csv(args.models)
    datasets = _split_csv(args.datasets)

    env_root = Path(args.env_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_results: list[RunResult] = []
    for ds in datasets:
        for model in models:
            print(f"[run] model={model} dataset={ds}")
            try:
                result = _run_one(
                    model=model,
                    dataset=ds,
                    horizon=args.horizon,
                    timeout_sec=args.timeout_sec,
                    env_root=env_root,
                    dry_run=args.dry_run,
                )
            except Exception as e:  # defensive
                result = RunResult(
                    model=model,
                    dataset=ds,
                    success=False,
                    elapsed_sec=0.0,
                    command="",
                    env="unknown",
                    error=str(e),
                    extra={},
                )
            run_results.append(result)

    metrics_rows, summary_rows = _rows_from_results(run_results)
    metrics_df = pd.DataFrame(metrics_rows)
    summary_df = pd.DataFrame(summary_rows)

    metrics_path = out_dir / "external_baseline_metrics_all.csv"
    summary_path = out_dir / "external_baseline_summary_all.csv"
    agg_summary_path = out_dir / "external_baseline_eval_summary_all.csv"

    metrics_df.to_csv(metrics_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    _summarize_metrics(metrics_df).to_csv(agg_summary_path, index=False)

    # Also emit per-dataset files to align with current repo conventions.
    for ds in datasets:
        mdf = metrics_df[metrics_df["dataset"] == ds].copy()
        sdf = summary_df[summary_df["dataset"] == ds].copy()
        mdf.to_csv(out_dir / f"external_baseline_metrics_{ds}.csv", index=False)
        sdf.to_csv(out_dir / f"external_baseline_summary_{ds}.csv", index=False)
        _summarize_metrics(mdf).to_csv(out_dir / f"external_baseline_eval_summary_{ds}.csv", index=False)

    print(
        json.dumps(
            {"metrics": str(metrics_path), "summary": str(summary_path), "eval_summary": str(agg_summary_path), "runs": len(run_results)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
