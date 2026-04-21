from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass
class DatasetBundle:
    name: str
    frame: pd.DataFrame
    target_col: str


def _canonicalize(frame: pd.DataFrame, dataset_name: str, target_col: Optional[str] = None) -> DatasetBundle:
    df = frame.copy()
    if "ds" not in df.columns:
        if "date" in df.columns:
            df = df.rename(columns={"date": "ds"})
        elif "timestamp" in df.columns:
            df = df.rename(columns={"timestamp": "ds"})
        else:
            df.insert(0, "ds", pd.RangeIndex(start=0, stop=len(df), step=1))

    if target_col is None:
        numeric_cols = [c for c in df.columns if c not in {"ds", "unique_id"} and pd.api.types.is_numeric_dtype(df[c])]
        if not numeric_cols:
            raise ValueError(f"No numeric target column found for {dataset_name}")
        target_col = numeric_cols[0]

    if target_col != "y":
        df = df.rename(columns={target_col: "y"})
    if "unique_id" not in df.columns:
        df.insert(0, "unique_id", dataset_name)

    return DatasetBundle(name=dataset_name, frame=df[["unique_id", "ds", "y"]], target_col="y")


def _load_long_horizon(dataset_name: str) -> Optional[DatasetBundle]:
    name_map = {
        "electricity": "ECL",
        "ecl": "ECL",
        "traffic": "TrafficL",
        "weather": "Weather",
        "ettm1": "ETTm1",
        "ettm2": "ETTm2",
        "etth1": "ETTh1",
        "etth2": "ETTh2",
    }
    group = name_map.get(dataset_name.lower())
    if group is None:
        return None

    try:
        from datasetsforecast.long_horizon import LongHorizon  # type: ignore

        Y_df, *_ = LongHorizon.load(directory="data/raw/long_horizon", group=group, cache=False)
        cols = {"ds": "ds", "y": "y", "unique_id": "unique_id"}
        if not all(c in Y_df.columns for c in cols):
            return _canonicalize(Y_df, dataset_name)
        return DatasetBundle(name=dataset_name, frame=Y_df[["unique_id", "ds", "y"]].copy(), target_col="y")
    except Exception:
        return None


def _load_ett(dataset_name: str, data_root: Path) -> Optional[DatasetBundle]:
    lower = dataset_name.lower()
    ett_map = {
        "ett": "ETTh1.csv",
        "etth1": "ETTh1.csv",
        "etth2": "ETTh2.csv",
        "ettm1": "ETTm1.csv",
        "ettm2": "ETTm2.csv",
    }
    fname = ett_map.get(lower)
    if fname is None:
        return None

    candidates = [
        data_root / fname,
        Path("external/ETDataset/ETT-small") / fname,
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p)
            target = "OT" if "OT" in df.columns else None
            return _canonicalize(df, dataset_name, target_col=target)
    return None


def _load_metr_la(data_root: Path) -> Optional[DatasetBundle]:
    candidates = [
        data_root / "metr-la.h5",
        Path("external/DCRNN/data/metr-la.h5"),
        Path("external/DCRNN/data/METR-LA/metr-la.h5"),
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_hdf(p)
            if isinstance(df, pd.Series):
                df = df.to_frame(name="y")
            if "y" not in df.columns:
                first_col = df.columns[0]
                df = df.rename(columns={first_col: "y"})
            df = df.reset_index().rename(columns={df.index.name or "index": "ds"})
            return _canonicalize(df, "metr_la", target_col="y")
    return None


def load_dataset(name: str, data_root: Path = Path("data/raw")) -> DatasetBundle:
    dataset_name = name.lower()

    ett_bundle = _load_ett(dataset_name, data_root)
    if ett_bundle is not None:
        return ett_bundle

    if dataset_name in {"metr_la", "metr-la"}:
        metr_bundle = _load_metr_la(data_root)
        if metr_bundle is not None:
            return metr_bundle

    long_horizon_bundle = _load_long_horizon(dataset_name)
    if long_horizon_bundle is not None:
        return long_horizon_bundle

    raise FileNotFoundError(
        f"Could not load dataset '{name}'. Supported direct sources: "
        "datasetsforecast.LongHorizon, ETDataset CSVs, DCRNN metr-la.h5, or data/raw/<name>.csv"
    )
