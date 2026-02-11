"""
Chunk aggregation and score fusion methods.

Provides various methods for:
1. Aggregating per-chunk similarities within a chunkset
2. Fusing chunkset-level and chunk-level scores
3. Adaptive weighting based on signal characteristics
"""

import numpy as np
from typing import Literal


# =====================================================================
# Chunk Aggregation Methods
# =====================================================================

def agg_max(chunk_sims: np.ndarray) -> float:
    """Original max aggregation - sensitive to outliers and chunkset size."""
    if len(chunk_sims) == 0:
        return 0.0
    return float(np.max(chunk_sims))


def agg_topk_mean(chunk_sims: np.ndarray, k: int = 3) -> float:
    """
    Top-k mean aggregation - robust max.
    Good when answers span multiple adjacent chunks, reduces one-off false spikes.
    """
    if len(chunk_sims) == 0:
        return 0.0
    k = min(k, len(chunk_sims))
    top_k = np.partition(chunk_sims, -k)[-k:]
    return float(np.mean(top_k))


def agg_percentile(chunk_sims: np.ndarray, p: float = 95.0) -> float:
    """
    Percentile aggregation - similar to max but less "lucky".
    p in {90, 95, 97} often works well.
    """
    if len(chunk_sims) == 0:
        return 0.0
    return float(np.percentile(chunk_sims, p))


def agg_softmax_pool(chunk_sims: np.ndarray, tau: float = 0.05) -> float:
    """
    Softmax pooling with temperature control.
    tau small -> behaves like max; tau large -> behaves like mean.
    tau in {0.03, 0.05, 0.08, 0.12} gives useful continuum.
    """
    if len(chunk_sims) == 0:
        return 0.0
    # Subtract max for numerical stability
    sims = chunk_sims - np.max(chunk_sims)
    weights = np.exp(sims / tau)
    weights = weights / np.sum(weights)
    return float(np.sum(weights * chunk_sims))


def agg_log_mean_exp(chunk_sims: np.ndarray, tau: float = 0.05) -> float:
    """
    Size-normalized log-sum-exp ("log-mean-exp").
    agg = tau * (log sum exp(sims/tau) - log n_chunks)

    This is the "right" fix for chunkset-size bias: smooth-max minus the
    expected uplift from having many draws.
    """
    if len(chunk_sims) == 0:
        return 0.0
    n = len(chunk_sims)
    # For numerical stability, subtract max before exp
    max_sim = np.max(chunk_sims)
    shifted = (chunk_sims - max_sim) / tau
    log_sum_exp = max_sim + tau * np.log(np.sum(np.exp(shifted)))
    return float(log_sum_exp - tau * np.log(n))


def agg_spike_score(chunk_sims: np.ndarray, reference: str = 'median') -> float:
    """
    Spike score - measures "needle-ness".
    High when there's a single standout chunk; helps separate
    "one chunk matches strongly" from "everything is moderately relevant".

    reference: 'median' or 'p75' (75th percentile)
    """
    if len(chunk_sims) == 0:
        return 0.0
    if len(chunk_sims) == 1:
        return float(chunk_sims[0])

    max_sim = np.max(chunk_sims)
    if reference == 'median':
        ref = np.median(chunk_sims)
    else:  # p75
        ref = np.percentile(chunk_sims, 75)
    return float(max_sim - ref)


def agg_max_adjusted(chunk_sims: np.ndarray, tau: float = 0.02) -> float:
    """
    Size-normalized hard max (original formula).
    max_adj = max(chunk_sims) - tau * log(n_chunks)

    Simple and effective fix for "big chunksets dominate" problem.
    """
    if len(chunk_sims) == 0:
        return 0.0
    n = len(chunk_sims)
    return float(np.max(chunk_sims) - tau * np.log(n))


def agg_smoothmax_adjusted(chunk_sims: np.ndarray, tau: float = 0.02) -> float:
    """
    Size-normalized smooth-max using log-sum-exp (smoothmax_adj).

    Formula: τ * (log(Σ exp(s_i/τ)) - log(n))

    This is the principled version of the "max - β*log(n)" formula:
    - Built-in size correction via the -τ*log(n) term
    - Smooth max is less brittle to single outliers than hard max
    - As τ→0, approaches max - τ*log(n)

    The -τ*log(n) correction falls out naturally from normalizing by n,
    which fixes the "lucky max" problem in a mathematically consistent way.
    """
    if len(chunk_sims) == 0:
        return 0.0
    n = len(chunk_sims)
    # For numerical stability, subtract max before exp
    max_sim = np.max(chunk_sims)
    shifted = (chunk_sims - max_sim) / tau
    log_sum_exp = max_sim + tau * np.log(np.sum(np.exp(shifted)))
    return float(log_sum_exp - tau * np.log(n))


# =====================================================================
# Score Fusion Methods
# =====================================================================

def fuse_linear(cs_sim: float, chunk_agg: float, w: float = 0.5) -> float:
    """
    Original linear fusion: w * cs_sim + (1-w) * chunk_agg
    """
    return w * cs_sim + (1 - w) * chunk_agg


def fuse_multiplicative(cs_sim: float, chunk_agg: float, alpha: float = 0.5,
                         a: float = 10.0, b: float = 0.5) -> float:
    """
    Multiplicative fusion after squashing to probabilities.
    score = p_cs^alpha * p_chunk^(1-alpha)

    Behaves like an AND: both signals must be decent.
    Great when one signal alone is noisy.

    a, b control sigmoid squashing: p = sigmoid(a * (s - b))
    """
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    p_cs = sigmoid(a * (cs_sim - b))
    p_chunk = sigmoid(a * (chunk_agg - b))
    return float(p_cs ** alpha * p_chunk ** (1 - alpha))


def fuse_relu_boost(cs_sim: float, chunk_agg: float, lam: float = 0.5) -> float:
    """
    Additive fusion with "only boost when it adds information".
    score = cs_sim + lambda * relu(chunk_agg - cs_sim)

    Chunk evidence can upgrade a chunkset beyond what its summary embedding
    suggests, but doesn't drag it down. Often improves recall without
    destroying coarse ranking.
    """
    boost = max(0.0, chunk_agg - cs_sim)
    return float(cs_sim + lam * boost)


# =====================================================================
# Adaptive Gating
# =====================================================================

def compute_gate_spike(cs_sim: float, chunk_sims: np.ndarray,
                       c0: float = 0.0, c1: float = 2.0, c2: float = 5.0) -> float:
    """
    Gate on spike-ness.
    spike = max - p75
    g = sigmoid(c0 + c1*cs_sim - c2*spike)

    High spike -> trust chunk evidence more (smaller g).
    Returns g in [0, 1] where g=1 means trust cs_sim fully.
    """
    if len(chunk_sims) == 0:
        return 0.5
    spike = float(np.max(chunk_sims) - np.percentile(chunk_sims, 75))
    logit = c0 + c1 * cs_sim - c2 * spike
    return float(1 / (1 + np.exp(-logit)))


def compute_gate_disagreement(cs_sim: float, chunk_agg: float,
                               d0: float = 0.0, d1: float = 5.0) -> float:
    """
    Gate on disagreement between cs_sim and chunk_agg.
    delta = chunk_agg - cs_sim
    g = sigmoid(d0 - d1*delta)

    If chunk-level says "much more relevant than chunkset embedding", let chunk win.
    """
    delta = chunk_agg - cs_sim
    logit = d0 - d1 * delta
    return float(1 / (1 + np.exp(-logit)))


def compute_gate_size(n_chunks: int, e0: float = 0.5, e1: float = 0.3) -> float:
    """
    Gate on chunkset size (controls max bias).
    g = sigmoid(e0 + e1*log(n_chunks))

    Bigger chunksets -> rely less on raw max unless you size-normalize.
    """
    logit = e0 + e1 * np.log(n_chunks)
    return float(1 / (1 + np.exp(-logit)))


def fuse_gated(cs_sim: float, chunk_agg: float, g: float) -> float:
    """
    Piecewise gated fusion.
    score = g * cs_sim + (1-g) * chunk_agg

    Even a crude gate can beat fixed w.
    """
    return g * cs_sim + (1 - g) * chunk_agg


# =====================================================================
# Rank-Level Fusion
# =====================================================================

def rrf_score(rank_a: int, rank_b: int, k: int = 60) -> float:
    """
    Reciprocal Rank Fusion score.
    rrf(x) = 1/(k + rankA(x)) + 1/(k + rankB(x))

    Scale-free, robust when two signals are complementary and differently calibrated.
    """
    return 1.0 / (k + rank_a) + 1.0 / (k + rank_b)


# =====================================================================
# Aggregation Method Registry
# =====================================================================

AGGREGATION_METHODS = {
    'max': agg_max,
    'topk2': lambda x: agg_topk_mean(x, k=2),
    'topk3': lambda x: agg_topk_mean(x, k=3),
    'topk5': lambda x: agg_topk_mean(x, k=5),
    'p90': lambda x: agg_percentile(x, p=90),
    'p95': lambda x: agg_percentile(x, p=95),
    'softmax_t03': lambda x: agg_softmax_pool(x, tau=0.03),
    'softmax_t05': lambda x: agg_softmax_pool(x, tau=0.05),
    'softmax_t08': lambda x: agg_softmax_pool(x, tau=0.08),
    'lme_t05': lambda x: agg_log_mean_exp(x, tau=0.05),
    'lme_t08': lambda x: agg_log_mean_exp(x, tau=0.08),
    'spike': lambda x: agg_spike_score(x, 'median'),
    'max_adj': lambda x: agg_max_adjusted(x, tau=0.02),
    'smoothmax_adj': lambda x: agg_smoothmax_adjusted(x, tau=0.02),
}


FUSION_METHODS = {
    'linear': fuse_linear,
    'multiplicative': fuse_multiplicative,
    'relu_boost': fuse_relu_boost,
    'gated': fuse_gated,
}
