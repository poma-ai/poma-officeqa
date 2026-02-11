"""
Retrieval functions: semantic ranking via cosine similarity.
"""

import numpy as np
from .models import EmbeddingResult


def hybrid_rank_matrix(
    semantic_vecs: np.ndarray,
    query_vecs: np.ndarray,
    documents: list[str],
    queries: list[str],
    top_k: int = 100,
    max_budget: int | None = None,
    avg_chunkset_tokens: int | None = None,
) -> np.ndarray:
    """Rank documents by semantic similarity to queries.

    Args:
        semantic_vecs: Document embeddings (n_docs, dims)
        query_vecs: Query embeddings (n_queries, dims)
        documents: Document texts (unused, kept for API compat)
        queries: Query texts (unused, kept for API compat)
        top_k: Number of results per query
        max_budget: If set, return all docs ranked
        avg_chunkset_tokens: Unused, kept for API compat

    Returns:
        Ranked indices matrix (n_queries, k)
    """
    n_queries = len(queries)
    n_docs = len(documents)

    if max_budget is not None:
        k = n_docs  # Rank all for budget-aware mode
    else:
        k = min(top_k, n_docs)

    ranked_matrix = np.zeros((n_queries, k), dtype=np.int32)

    # Semantic similarities (float64 for BLAS stability)
    sem_f64 = semantic_vecs.astype(np.float64)
    q_f64 = query_vecs.astype(np.float64)
    sem_f64 = np.nan_to_num(sem_f64, nan=0.0)
    q_f64 = np.nan_to_num(q_f64, nan=0.0)

    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        sims = np.dot(sem_f64, q_f64.T)

    n_bad = np.sum(~np.isfinite(sims))
    if n_bad > 0:
        sims = np.nan_to_num(sims, nan=0.0, posinf=1.0, neginf=-1.0)
    sims = sims.astype(np.float32)

    for qi in range(n_queries):
        ranked = np.argsort(-sims[:, qi]).tolist()
        ranked_matrix[qi, :k] = ranked[:k]

    return ranked_matrix


def retrieve_by_maxsim(
    emb_result: EmbeddingResult,
    q_vecs: np.ndarray,
    k: int,
    cs_texts: list[str] = None,
    q_texts: list[str] = None,
    max_budget: int | None = None,
    avg_chunkset_tokens: int | None = None,
) -> np.ndarray:
    """Retrieve by max similarity across prefix/leaf embeddings.

    Args:
        emb_result: EmbeddingResult with prefix/leaf vectors
        q_vecs: Query embeddings
        k: Number of results per query
        cs_texts: Chunkset texts (unused, kept for API compat)
        q_texts: Query texts (unused, kept for API compat)
        max_budget: If set, return all chunksets ranked
        avg_chunkset_tokens: Unused, kept for API compat

    Returns:
        Ranked chunkset indices (n_queries, k)
    """
    P = emb_result.chunkset_prefix_vecs.astype(np.float64)
    L = emb_result.chunkset_leaf_vecs.astype(np.float64)
    Q = q_vecs.astype(np.float64)

    P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    L = np.nan_to_num(L, nan=0.0, posinf=0.0, neginf=0.0)
    Q = np.nan_to_num(Q, nan=0.0, posinf=0.0, neginf=0.0)

    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        P_scores = P @ Q.T
        L_scores = L @ Q.T

    p_bad = np.sum(~np.isfinite(P_scores))
    l_bad = np.sum(~np.isfinite(L_scores))
    if p_bad > 0 or l_bad > 0:
        P_scores = np.nan_to_num(P_scores, nan=0.0, posinf=1.0, neginf=-1.0)
        L_scores = np.nan_to_num(L_scores, nan=0.0, posinf=1.0, neginf=-1.0)

    maxsim_scores = np.maximum(P_scores, L_scores).astype(np.float32)
    n_q = Q.shape[0]
    n_cs = maxsim_scores.shape[0]

    if max_budget is not None:
        k = n_cs
    else:
        k = min(k, n_cs)

    out = np.zeros((n_q, k), dtype=np.int32)
    for qi in range(n_q):
        out[qi] = np.argsort(-maxsim_scores[:, qi])[:k]

    return out
