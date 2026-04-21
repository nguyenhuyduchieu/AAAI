from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors


def bootstrap_corr_ci(
    x: np.ndarray,
    y: np.ndarray,
    corr_type: str,
    n_resamples: int,
    seed: int,
) -> tuple[float, float, float, float]:
    if corr_type == "spearman":
        corr = stats.spearmanr(x, y)
        stat = float(corr.correlation)
        pval = float(corr.pvalue)
    elif corr_type == "pearson":
        corr = stats.pearsonr(x, y)
        stat = float(corr.statistic)
        pval = float(corr.pvalue)
    else:
        raise ValueError(f"Unsupported corr_type: {corr_type}")
    if not np.isfinite(stat):
        return 0.0, 0.0, 0.0, 1.0

    n = len(x)
    if n < 5:
        return stat, stat, stat, pval

    rng = np.random.default_rng(seed)
    boots = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        xi, yi = x[idx], y[idx]
        if corr_type == "spearman":
            boots[i] = stats.spearmanr(xi, yi).correlation
        else:
            boots[i] = stats.pearsonr(xi, yi).statistic
    boots = np.nan_to_num(boots, nan=0.0)
    ci_low, ci_high = np.quantile(boots, [0.025, 0.975])
    return stat, float(ci_low), float(ci_high), pval


def _sample_pairs(n: int, max_pairs: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if n < 2:
        return np.array([], dtype=int), np.array([], dtype=int)
    all_pairs = n * (n - 1) // 2
    if all_pairs <= max_pairs:
        return np.triu_indices(n, k=1)
    rng = np.random.default_rng(seed)
    i = rng.integers(0, n, size=max_pairs)
    j = rng.integers(0, n, size=max_pairs)
    mask = i != j
    return i[mask], j[mask]


def compute_fss(
    embeddings: np.ndarray,
    preds: np.ndarray,
    corr_type: str,
    max_pairs: int,
    n_resamples: int,
    seed: int,
) -> tuple[float, float, float, float, float]:
    i, j = _sample_pairs(len(embeddings), max_pairs=max_pairs, seed=seed)
    if len(i) == 0:
        return 0.0, 0.0, 0.0, np.nan, 0.0
    dz = np.linalg.norm(embeddings[i] - embeddings[j], axis=1)
    dy = np.abs(preds[i] - preds[j])
    raw, ci_low, ci_high, pval = bootstrap_corr_ci(dz, dy, corr_type=corr_type, n_resamples=n_resamples, seed=seed)
    return -raw, -ci_high, -ci_low, pval, raw


def compute_pds(
    embeddings: np.ndarray,
    errors: np.ndarray,
    k: int,
    corr_type: str,
    n_resamples: int,
    seed: int,
) -> tuple[float, float, float, float, float]:
    k_eff = max(2, min(k, len(embeddings) - 1))
    nbrs = NearestNeighbors(n_neighbors=k_eff + 1, metric="euclidean")
    nbrs.fit(embeddings)
    distances, _ = nbrs.kneighbors(embeddings)
    rho = distances[:, 1:].mean(axis=1)
    raw, ci_low, ci_high, pval = bootstrap_corr_ci(rho, errors, corr_type=corr_type, n_resamples=n_resamples, seed=seed)
    return -raw, -ci_high, -ci_low, pval, raw


def compute_ltc(embeddings: np.ndarray, starts: np.ndarray) -> tuple[float, float]:
    order = np.argsort(starts)
    z = embeddings[order]
    if len(z) < 3:
        return 0.0, 0.0
    delta = z[1:] - z[:-1]
    step = np.linalg.norm(delta, axis=1)
    v1 = delta[:-1]
    v2 = delta[1:]
    denom = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
    denom = np.clip(denom, 1e-8, None)
    directional = np.mean(np.sum(v1 * v2, axis=1) / denom)
    return float(np.mean(step)), float(directional)


def compute_lls(embeddings: np.ndarray, k: int) -> float:
    k_eff = max(3, min(k, len(embeddings) - 1))
    nbrs = NearestNeighbors(n_neighbors=k_eff + 1, metric="euclidean")
    nbrs.fit(embeddings)
    _, idx = nbrs.kneighbors(embeddings)
    ratios: list[float] = []
    for neighbors in idx:
        local = embeddings[neighbors[1:]]
        if local.shape[0] < 3:
            continue
        pca = PCA(n_components=min(local.shape[0], local.shape[1]))
        pca.fit(local)
        ev = pca.explained_variance_
        ratios.append(float(ev[0] / (np.sum(ev) + 1e-8)))
    if not ratios:
        return 0.0
    return float(np.mean(ratios))


def compute_mcs(train_embeddings: np.ndarray, test_embeddings: np.ndarray, k: int, quantile: float) -> float:
    if len(train_embeddings) < 3 or len(test_embeddings) == 0:
        return 0.0
    k_eff = max(2, min(k, len(train_embeddings) - 1))
    nbrs_train = NearestNeighbors(n_neighbors=k_eff + 1, metric="euclidean")
    nbrs_train.fit(train_embeddings)
    d_train, _ = nbrs_train.kneighbors(train_embeddings)
    support_radius = float(np.quantile(d_train[:, 1:].mean(axis=1), quantile))

    nbrs_test = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nbrs_test.fit(train_embeddings)
    d_test, _ = nbrs_test.kneighbors(test_embeddings)
    covered = (d_test[:, 0] <= support_radius).astype(float)
    return float(np.mean(covered))


def compute_tcs(embeddings: np.ndarray, transformed_embeddings: np.ndarray) -> float:
    num = np.sum(embeddings * transformed_embeddings, axis=1)
    den = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(transformed_embeddings, axis=1)
    den = np.clip(den, 1e-8, None)
    return float(np.mean(num / den))


def compute_css(
    embeddings: np.ndarray,
    labels: np.ndarray,
    seed: int,
    n_permutations: int = 100,
) -> tuple[float, float, float]:
    if len(np.unique(labels)) < 2:
        return 0.0, 0.0, 1.0
    x_train, x_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=0.2, random_state=seed, stratify=labels
    )
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    bal_acc = float(balanced_accuracy_score(y_test, pred))
    margin = float(np.mean(np.max(clf.predict_proba(x_test), axis=1)))

    rng = np.random.default_rng(seed)
    perm_scores = np.empty(n_permutations, dtype=np.float64)
    for i in range(n_permutations):
        yp = rng.permutation(labels)
        xp_train, xp_test, yp_train, yp_test = train_test_split(
            embeddings, yp, test_size=0.2, random_state=seed + i, stratify=yp
        )
        pclf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed + i)
        pclf.fit(xp_train, yp_train)
        ppred = pclf.predict(xp_test)
        perm_scores[i] = balanced_accuracy_score(yp_test, ppred)
    pvalue = float((np.sum(perm_scores >= bal_acc) + 1) / (len(perm_scores) + 1))
    return bal_acc, margin, pvalue


def primitive_labels_from_stl_features(stl_features: np.ndarray) -> dict[str, np.ndarray]:
    # Columns follow regime_labels._window_stl_features:
    # [seasonal_strength, trend_strength, resid_volatility, resid_to_total, slope_scale, trend_delta, seasonal_amplitude]
    seasonality = (stl_features[:, 0] > np.median(stl_features[:, 0])).astype(int)
    trend = (stl_features[:, 1] > np.median(stl_features[:, 1])).astype(int)
    volatility = (stl_features[:, 2] > np.median(stl_features[:, 2])).astype(int)
    return {"seasonality": seasonality, "trend": trend, "volatility": volatility}

