from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def phase_mod_labels(n_samples: int, modulus: int = 8) -> np.ndarray:
    if n_samples <= 0:
        return np.zeros((0,), dtype=int)
    return (np.arange(n_samples) % int(modulus)).astype(int)


def _window_stl_features(window: np.ndarray, period: int) -> np.ndarray:
    from statsmodels.tsa.seasonal import STL

    y = np.asarray(window, dtype=np.float64)
    t = np.arange(len(y), dtype=np.float64)
    stl = STL(y, period=period, robust=True).fit()

    trend = stl.trend
    seasonal = stl.seasonal
    resid = stl.resid

    eps = 1e-8
    total_var = np.var(y) + eps
    resid_var = np.var(resid)
    seasonal_var = np.var(seasonal)
    trend_var = np.var(trend)

    # Standard STL-derived strengths from decomposition variances.
    seasonal_strength = max(0.0, 1.0 - resid_var / (resid_var + seasonal_var + eps))
    trend_strength = max(0.0, 1.0 - resid_var / (resid_var + trend_var + eps))
    resid_volatility = float(np.std(resid))
    resid_to_total = float(resid_var / total_var)

    slope, _ = np.polyfit(t, trend, deg=1)
    slope_scale = float(slope / (np.std(trend) + eps))
    trend_delta = float((trend[-1] - trend[0]) / (np.std(y) + eps))
    seasonal_amplitude = float((np.percentile(seasonal, 95) - np.percentile(seasonal, 5)) / (np.std(y) + eps))

    return np.array(
        [
            seasonal_strength,
            trend_strength,
            resid_volatility,
            resid_to_total,
            slope_scale,
            trend_delta,
            seasonal_amplitude,
        ],
        dtype=np.float64,
    )


def stl_regime_labels(
    windows: np.ndarray,
    period: int,
    n_clusters: int = 8,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    if len(windows) == 0:
        return np.zeros((0,), dtype=int), np.zeros((0, 7), dtype=np.float64)

    if period < 2:
        raise ValueError("STL period must be >= 2.")

    feature_rows = np.stack([_window_stl_features(w, period=period) for w in windows], axis=0)
    feature_rows = np.nan_to_num(feature_rows, nan=0.0, posinf=1e6, neginf=-1e6)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature_rows)

    k = int(max(2, min(n_clusters, len(scaled))))
    labels = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit_predict(scaled)
    return labels.astype(int), feature_rows
