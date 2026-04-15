from __future__ import annotations

import numpy as np
from sklearn.metrics import silhouette_score


def _finite_rows(embeddings: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(embeddings).all(axis=1)
    return embeddings[mask], labels[mask]


def compute_lcs(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """Latent Cohesion Score based on silhouette."""
    emb = np.nan_to_num(embeddings, nan=0.0, posinf=1e6, neginf=-1e6)
    emb, lab = _finite_rows(emb, labels)
    if len(emb) < 2 or len(np.unique(lab)) < 2:
        return 0.0
    return float(silhouette_score(emb, lab))


def compute_tsi(clean_embeddings: np.ndarray, perturbed_embeddings: np.ndarray) -> float:
    """Temporal Stability Index: cosine-like similarity over trajectories."""
    clean = np.nan_to_num(clean_embeddings, nan=0.0, posinf=1e6, neginf=-1e6)
    perturbed = np.nan_to_num(perturbed_embeddings, nan=0.0, posinf=1e6, neginf=-1e6)
    num = np.sum(clean * perturbed, axis=1)
    den = np.linalg.norm(clean, axis=1) * np.linalg.norm(perturbed, axis=1)
    den = np.clip(den, 1e-8, None)
    return float(np.mean(num / den))


def compute_gsp(clean_score: float, perturbed_score: float) -> float:
    """Generalized Stability under Perturbation."""
    return float(clean_score - perturbed_score)
