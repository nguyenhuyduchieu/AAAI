from __future__ import annotations

import numpy as np
import tsaug


def _sanitize(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)


def apply_noise(x: np.ndarray, scale: float = 0.02) -> np.ndarray:
    return _sanitize(tsaug.AddNoise(scale=scale).augment(x))


def apply_masking(x: np.ndarray, p: float = 0.05) -> np.ndarray:
    return _sanitize(tsaug.Dropout(p=p).augment(x))


def apply_time_warp(x: np.ndarray, n_speed_change: int = 3, max_speed_ratio: float = 2.0) -> np.ndarray:
    augmenter = tsaug.TimeWarp(n_speed_change=n_speed_change, max_speed_ratio=max_speed_ratio)
    try:
        return _sanitize(augmenter.augment(x))
    except ValueError as exc:
        # tsaug TimeWarp can fail on some shapes/value ranges with
        # "x must be strictly increasing sequence". Fallback keeps
        # perturbation protocol running instead of aborting the pipeline.
        print(f"[perturbation] TimeWarp failed ({exc}); fallback to Drift.")
        return _sanitize(tsaug.Drift(max_drift=0.1, n_drift_points=3).augment(x))
