from __future__ import annotations

import numpy as np


def pool_sequence_embedding(token_embeddings: np.ndarray, mode: str = "mean") -> np.ndarray:
    if mode == "mean":
        return token_embeddings.mean(axis=1)
    if mode == "last":
        return token_embeddings[:, -1, :]
    raise ValueError(f"Unsupported pooling mode: {mode}")
