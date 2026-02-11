"""
Embedding utility functions.

Note: For embeddings, use bench/vectors.py which uses the new .npz format.
Text normalization for embedding (normalize_for_embedding) now lives in poma-core/utils.py
and is delivered as the `to_embed` field on chunks and chunksets.
"""

import math

import numpy as np


def norm(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize vector to unit length."""
    v = np.asarray(v, dtype=np.float32)
    if not np.isfinite(v).all():
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    n = float(np.linalg.norm(v))
    if not math.isfinite(n) or n < eps:
        return np.zeros_like(v, dtype=np.float32)
    return (v / n).astype(np.float32, copy=False)
