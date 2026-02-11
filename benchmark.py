#!/usr/bin/env python3
"""
Benchmark with INDEX-BASED evaluation.

Checks if golden chunk INDICES from found_in are retrieved, NOT if needle values appear in text.
This is the correct approach - we verified the indices contain the needles, now just check retrieval.
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from bench import (
    EXTENDED_K,
    load_poma_corpus, load_raw_poma_data,
    load_naive_chunks_from_txt, load_unstructured_elements, load_officeqa,
    hybrid_rank_matrix,
    get_valid_uids,
    # Aggregation and fusion methods
    AGGREGATION_METHODS, FUSION_METHODS,
    fuse_linear,
)
from bench import config as bench_config
from bench.config import MAX_BUDGET_OVERAGE
from bench.context import n_tokens, build_cheatsheet_from_indices
from bench.evaluation_index import get_golden_indices, set_corpus_offsets

script_dir = Path(__file__).parent


# =====================================================================
# Micro-caching for efficient adaptive search
# =====================================================================

MICRO_CACHE_DIR = script_dir / "results" / "micro_results_idx"

# Global scoring config hash - set in main() based on args
_SCORING_CONFIG_HASH = None


def compute_scoring_hash(weight: float, agg: str, fusion: str, zscore: bool) -> str:
    """Compute a short hash of scoring parameters + vector file mtimes for cache invalidation.

    Captures:
    - Scoring params (weight, agg, fusion, zscore)
    - Vector file modification times (detects re-embedding)
    """
    import hashlib

    # Scoring params
    config_str = f"w{weight}_agg{agg}_fus{fusion}_z{int(not zscore)}"

    # Vector file mtimes (detect re-embedding)
    vectors_dir = script_dir / "vectors"
    for npz_file in ["questions.npz", "poma_chunksets.npz", "poma_chunks.npz"]:
        fpath = vectors_dir / npz_file
        if fpath.exists():
            config_str += f"_{npz_file}:{int(fpath.stat().st_mtime)}"

    return hashlib.md5(config_str.encode()).hexdigest()[:8]


def get_micro_cache_path(method: str, uid: str, budget: int) -> Path:
    # Include scoring config hash to auto-invalidate on param changes
    if _SCORING_CONFIG_HASH:
        return MICRO_CACHE_DIR / f"{method}_{_SCORING_CONFIG_HASH}_{uid}_{budget}.json"
    return MICRO_CACHE_DIR / f"{method}_{uid}_{budget}.json"


def load_micro_cache(method: str, uid: str, budget: int) -> dict | None:
    path = get_micro_cache_path(method, uid, budget)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def save_micro_cache(method: str, uid: str, budget: int, result: dict):
    MICRO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = get_micro_cache_path(method, uid, budget)
    with open(path, 'w') as f:
        json.dump(result, f, indent=2)


# =====================================================================
# Evaluation functions with proper cheatsheet deduplication
# =====================================================================

def evaluate_poma_at_budget(method, uid, budget, ranked, chunksets, raw_data):
    """
    Evaluate single POMA question at given budget.
    
    Two-phase approach for efficiency:
    1. Use raw token counting to find which chunksets fit in budget (fast)
    2. Build cheatsheet ONCE at the end to get actual deduplicated token count
    """
    from bench.context import build_cheatsheet_from_indices
    
    golden_indices = set(get_golden_indices(uid, 'poma'))
    
    if not golden_indices:
        return {"found": False, "reason": "no_golden", "n_found": 0, "n_total": 0}
    
    # Phase 1: Use RAW tokens to determine which chunksets fit in budget (fast)
    retrieved_indices = []
    raw_tokens = 0
    
    for idx in ranked:
        if idx < 0 or idx >= len(chunksets):
            continue
        cs = chunksets[idx]
        cs_raw_tokens = n_tokens(cs.contents)
        if raw_tokens + cs_raw_tokens > budget + MAX_BUDGET_OVERAGE:
            break
        retrieved_indices.append(idx)
        raw_tokens += cs_raw_tokens
    
    retrieved_set = set(retrieved_indices)
    n_found = len(golden_indices & retrieved_set)
    
    if golden_indices <= retrieved_set:
        # Phase 2: Build cheatsheet ONCE to get actual deduplicated token count
        try:
            context, cheatsheet_tokens = build_cheatsheet_from_indices(
                retrieved_indices, chunksets, raw_data
            )
        except Exception as e:
            # Fallback to raw tokens if cheatsheet fails
            cheatsheet_tokens = raw_tokens
        
        return {
            "found": True,
            "reason": f"indices:{len(golden_indices)}/{len(golden_indices)}",
            "n_found": len(golden_indices),
            "n_total": len(golden_indices),
            "tokens_used": cheatsheet_tokens,
            "raw_tokens": raw_tokens,
            "n_chunksets": len(retrieved_indices),
        }
    else:
        return {
            "found": False,
            "reason": f"indices_missing:{n_found}/{len(golden_indices)}",
            "n_found": n_found,
            "n_total": len(golden_indices),
            "tokens_used": raw_tokens,
        }


def evaluate_naive_at_budget(method, uid, budget, ranked, chunks, mode):
    """Evaluate single naive question at given budget."""
    golden_indices = set(get_golden_indices(uid, mode))
    
    if not golden_indices:
        return {"found": False, "reason": "no_golden", "n_found": 0, "n_total": 0}
    
    retrieved_indices = set()
    current_tokens = 0
    
    for idx in ranked:
        if idx < 0 or idx >= len(chunks):
            continue
        chunk = chunks[idx]
        chunk_tokens = n_tokens(chunk.text)
        if current_tokens + chunk_tokens > budget + MAX_BUDGET_OVERAGE:
            break
        retrieved_indices.add(idx)
        current_tokens += chunk_tokens
    
    n_found = len(golden_indices & retrieved_indices)
    
    if golden_indices <= retrieved_indices:
        return {
            "found": True,
            "reason": f"indices:{len(golden_indices)}/{len(golden_indices)}",
            "n_found": len(golden_indices),
            "n_total": len(golden_indices),
            "tokens_used": current_tokens,
        }
    else:
        return {
            "found": False,
            "reason": f"indices_missing:{n_found}/{len(golden_indices)}",
            "n_found": n_found,
            "n_total": len(golden_indices),
            "tokens_used": current_tokens,
        }


# =====================================================================
# Adaptive search: exponential probe up, then binary search down
# =====================================================================

def run_adaptive_search(method, questions, ranked_matrix, eval_fn, max_budget, initial_step, min_step):
    """
    Smart adaptive search: start low, exponentially increase until found, then binary search down.

    Strategy:
    1. Probe at initial_step (e.g., 50K)
    2. If not found, double budget until found or max_budget reached
    3. Once found, binary search between last-fail and first-success for exact minimum

    This is much faster than probing at max_budget first, since most questions
    are found at low budgets.
    """
    print(f"\n[{method}] Evaluating {len(questions)} questions (adaptive search)...")

    n_found = 0
    for qi, q in enumerate(questions):
        ranked = ranked_matrix[qi, :].tolist()

        # Phase 1: Exponential probe to find a budget that works
        probe_budget = initial_step
        last_fail = 0
        first_success = None

        while probe_budget <= max_budget:
            cached = load_micro_cache(method, q.uid, probe_budget)
            if cached:
                result = cached
            else:
                result = eval_fn(method, q.uid, probe_budget, ranked)
                result['timestamp'] = datetime.now().isoformat()
                result['method'] = method
                result['uid'] = q.uid
                result['budget'] = probe_budget
                save_micro_cache(method, q.uid, probe_budget, result)

            if result.get('found'):
                first_success = probe_budget
                break
            else:
                last_fail = probe_budget
                probe_budget = min(probe_budget * 2, max_budget) if probe_budget < max_budget else max_budget + 1

        if first_success is None:
            print(f"    [{q.uid}] => NOT FOUND @ {max_budget:,} budget")
            continue

        # Phase 2: Binary search between last_fail and first_success for exact minimum
        found_budget = result.get('tokens_used', first_success)
        low, high = last_fail, first_success

        while high - low > min_step:
            mid = (low + high) // 2
            mid = (mid // min_step) * min_step  # Round to step
            if mid <= low:
                mid = low + min_step

            cached = load_micro_cache(method, q.uid, mid)
            if cached:
                result = cached
            else:
                result = eval_fn(method, q.uid, mid, ranked)
                result['timestamp'] = datetime.now().isoformat()
                result['method'] = method
                result['uid'] = q.uid
                result['budget'] = mid
                save_micro_cache(method, q.uid, mid, result)

            if result.get('found'):
                found_budget = result.get('tokens_used', mid)
                high = mid
            else:
                low = mid + min_step

        print(f"    [{q.uid}] => FOUND @ {found_budget:,} tokens")
        n_found += 1

    print(f"  Summary: {n_found}/{len(questions)} questions found")


# =====================================================================
# Aggregate results
# =====================================================================

def aggregate_results(method, uids, report_budgets):
    """Aggregate cached results for a method."""
    results = {}
    n_answered = 0
    min_budgets = {}
    
    for uid in uids:
        # Find minimum budget where found=True
        best = None
        for budget in range(5000, 4000001, 5000):
            cached = load_micro_cache(method, uid, budget)
            if cached and cached.get('found'):
                tokens_used = cached.get('tokens_used', budget)
                if best is None or tokens_used < best:
                    best = tokens_used
        
        if best is not None:
            n_answered += 1
            min_budgets[uid] = best
            results[uid] = {"min_budget": best, "passed": True}
        else:
            results[uid] = {"min_budget": None, "passed": False}
    
    # Accuracy curve
    accuracy_curve = {}
    for budget in report_budgets:
        n_at_budget = sum(1 for mb in min_budgets.values() if mb <= budget + MAX_BUDGET_OVERAGE)
        accuracy_curve[budget] = n_at_budget / len(uids) if uids else 0.0
    
    return {
        "n_answered": n_answered,
        "n_total": len(uids),
        "questions": results,
        "accuracy_curve": accuracy_curve,
    }


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Index-based benchmark")
    parser.add_argument("--method", type=str, help="Run single method")
    parser.add_argument("--max-budget", type=int, default=2000000,
                        help="Max token budget per question. Default=2M (sufficient for 100%% recall on all methods)")
    parser.add_argument("--initial-step", type=int, default=50000)
    parser.add_argument("--min-step", type=int, default=5000)
    parser.add_argument("--weight", "-w", type=float, default=0.65,
                        help="Weight for chunkset sim (1-w for chunk sim). Default=0.65 (optimized)")
    parser.add_argument("--agg", type=str, default="max",
                        choices=list(AGGREGATION_METHODS.keys()),
                        help="Chunk aggregation method. Default=max (optimized)")
    parser.add_argument("--fusion", type=str, default="linear",
                        choices=list(FUSION_METHODS.keys()),
                        help="Score fusion method. Default=linear")
    parser.add_argument("--zscore", action="store_true",
                        help="Enable per-query z-scoring (default: disabled, use raw scores)")
    args = parser.parse_args()

    # Set scoring config hash for cache invalidation
    global _SCORING_CONFIG_HASH
    _SCORING_CONFIG_HASH = compute_scoring_hash(args.weight, args.agg, args.fusion, args.zscore)
    print(f"Scoring config hash: {_SCORING_CONFIG_HASH}")

    # Load corpora
    print("Loading corpora...")
    poma_dir = str(script_dir / "data" / "poma")
    officeqa_path = str(script_dir / "data" / "officeqa.csv")
    
    chunksets, filename_to_docid = load_poma_corpus(poma_dir)
    raw_data = load_raw_poma_data(poma_dir, filename_to_docid)
    naive_chunks_txt = load_naive_chunks_from_txt(poma_dir, filename_to_docid)
    unstructured_elements = load_unstructured_elements(filename_to_docid)
    questions = load_officeqa(officeqa_path, filename_to_docid)
    
    # Set up corpus offsets for index-based evaluation
    # The evidence file uses doc_name -> local_id format
    # We need to map doc_name to global offset in the corpus lists
    
    # POMA offsets (by doc_name, sorted)
    poma_offsets = {}
    current_offset = 0
    for doc_name in sorted(filename_to_docid.keys()):
        poma_offsets[doc_name] = current_offset
        doc_id = filename_to_docid[doc_name]
        n_cs = sum(1 for cs in chunksets if cs.doc_id == doc_id)
        current_offset += n_cs
    set_corpus_offsets('poma', poma_offsets)
    
    # DB_TXT offsets - chunks are ordered by doc_id (which maps to doc_name)
    # Build reverse mapping: doc_id -> doc_name
    docid_to_name = {v: k for k, v in filename_to_docid.items()}
    
    txt_offsets = {}
    current_offset = 0
    for doc_name in sorted(filename_to_docid.keys()):
        txt_offsets[doc_name] = current_offset
        doc_id = filename_to_docid[doc_name]
        n_chunks = sum(1 for c in naive_chunks_txt if c.doc_id == doc_id)
        current_offset += n_chunks
    set_corpus_offsets('db_txt', txt_offsets)
    
    # US offsets
    us_offsets = {}
    current_offset = 0
    for doc_name in sorted(filename_to_docid.keys()):
        us_offsets[doc_name] = current_offset
        doc_id = filename_to_docid[doc_name]
        n_chunks = sum(1 for c in unstructured_elements if c.doc_id == doc_id)
        current_offset += n_chunks
    set_corpus_offsets('us', us_offsets)
    
    # Get valid UIDs
    uids = get_valid_uids()
    questions = [q for q in questions if q.uid in uids]
    q_texts = [q.question for q in questions]
    
    print(f"Loaded {len(chunksets)} chunksets, {len(naive_chunks_txt)} txt chunks, {len(unstructured_elements)} US elements")
    print(f"Evaluating {len(questions)} questions")
    
    # Retrieval params for reporting
    retrieval_params = {
        "extended_k": EXTENDED_K,
    }
    
    # Methods - renamed for clarity
    # poma_openai_mixed: balanced scoring (w=0.5 chunkset + 0.5 chunk with max_adj)
    # poma_openai_chunksets: pure chunkset similarity (w=1.0, legacy)
    if args.weight == 1.0:
        poma_method_name = 'poma_openai_chunksets'
    else:
        poma_method_name = 'poma_openai_mixed'

    all_methods = ['poma_openai_chunksets', poma_method_name, 'databricks_rcs_openai',
                   'unstructured_openai']
    # Deduplicate if same
    all_methods = list(dict.fromkeys(all_methods))

    if args.method:
        if args.method in ['poma_balanced', 'poma_openai_mixed', 'mixed']:
            all_methods = ['poma_openai_mixed']
        elif args.method in ['poma_openai', 'poma_openai_chunksets', 'chunksets']:
            all_methods = ['poma_openai_chunksets']
        else:
            all_methods = [args.method]
    
    for method in all_methods:
        print(f"\n{'='*60}")
        print(f"Method: {method}")
        print(f"{'='*60}")
        
        if method == 'poma_openai_chunksets':
            from bench.vectors import load_vectors

            # Load pre-computed embeddings from .npz files
            try:
                cs_emb, cs_ids, _ = load_vectors("poma_chunksets")
                q_emb_all, q_ids, _ = load_vectors("questions")
            except FileNotFoundError as e:
                print(f"  ERROR: Vector file not found: {e}")
                print("  Run: python embed_all.py")
                continue

            # Build question UID to embedding index mapping
            uid_to_q_idx = {uid: idx for idx, uid in enumerate(q_ids)}

            # Get embeddings for just the questions we're evaluating
            q_emb = np.array([q_emb_all[uid_to_q_idx[q.uid]] for q in questions])

            cs_texts = [cs.contents for cs in chunksets]
            ranked_matrix = hybrid_rank_matrix(cs_emb, q_emb, cs_texts, q_texts, top_k=EXTENDED_K)

            def eval_fn(m, u, b, r):
                return evaluate_poma_at_budget(m, u, b, r, chunksets, raw_data)

        elif method == 'poma_openai_mixed':
            from numpy.linalg import norm as np_norm
            from bench.vectors import load_vectors

            # Load pre-computed embeddings from .npz files
            try:
                cs_emb, cs_ids, _ = load_vectors("poma_chunksets")
                chunk_emb, chunk_ids, chunk_meta = load_vectors("poma_chunks")
                chunk_map = json.loads(chunk_meta.get("chunk_map", "{}"))
            except FileNotFoundError as e:
                print(f"  ERROR: Vector file not found: {e}")
                print("  Run: python embed_all.py")
                continue

            # If chunk_map is empty, rebuild it from raw_data
            if not chunk_map:
                print("  Rebuilding chunk_map from raw_data...")
                # Build reverse index: chunk_id -> embedding_index
                # Use FIRST occurrence only (handles duplicate embeddings)
                chunk_id_to_emb_idx = {}
                for idx, cid in enumerate(chunk_ids):
                    if cid not in chunk_id_to_emb_idx:
                        chunk_id_to_emb_idx[cid] = idx
                print(f"  Built chunk_id lookup with {len(chunk_id_to_emb_idx)} unique entries from {len(chunk_ids)} total")

                # Build mapping from runtime doc_id to embedding doc_id
                # Both are in sorted filename order, but may use different indices
                # Extract unique doc indices from chunk_ids in order of first occurrence
                seen_emb_doc_ids = []
                for cid in chunk_ids:
                    emb_doc_id = cid.split(':')[0]
                    if emb_doc_id not in seen_emb_doc_ids:
                        seen_emb_doc_ids.append(emb_doc_id)

                # Runtime doc_ids are "0", "1", "2", ... in sorted filename order
                # Map runtime doc_id to embedding doc_id
                runtime_to_emb_doc_id = {str(i): seen_emb_doc_ids[i] for i in range(len(seen_emb_doc_ids))}
                print(f"  Doc ID mapping: runtime -> embedding: {runtime_to_emb_doc_id}")

                # For each chunkset, look up its chunks from raw_data
                for cs in chunksets:
                    doc_id = cs.doc_id  # Runtime doc_id like "0", "1", etc.
                    if doc_id not in raw_data:
                        continue

                    # Get the corresponding embedding doc_id
                    emb_doc_id = runtime_to_emb_doc_id.get(doc_id, doc_id)

                    chunks_list, chunksets_raw = raw_data[doc_id]
                    if cs.chunkset_id < len(chunksets_raw):
                        cs_raw = chunksets_raw[cs.chunkset_id]
                        cs_chunk_indices = cs_raw.get("chunks", [])

                        emb_indices = []
                        for chunk_idx in cs_chunk_indices:
                            # Use embedding doc_id to look up chunks
                            chunk_id = f"{emb_doc_id}:chunk:{chunk_idx}"
                            if chunk_id in chunk_id_to_emb_idx:
                                emb_indices.append(chunk_id_to_emb_idx[chunk_id])

                        if emb_indices:
                            cs_key = f"{doc_id}:chunkset:{cs.chunkset_id}"
                            chunk_map[cs_key] = emb_indices

                print(f"  Rebuilt chunk_map with {len(chunk_map)} entries")

            # Load question embeddings
            q_emb, q_ids, _ = load_vectors("questions")

            # Build question UID to embedding index mapping
            uid_to_q_idx = {uid: idx for idx, uid in enumerate(q_ids)}

            # Get the selected aggregation function
            agg_fn = AGGREGATION_METHODS[args.agg]
            fusion_fn = FUSION_METHODS[args.fusion]

            print(f"  Using weight w={args.weight}, agg={args.agg}, fusion={args.fusion}")
            print(f"  Loaded {len(cs_emb)} chunkset embeddings, {len(chunk_emb)} chunk embeddings")

            n_queries = len(questions)
            n_chunksets = len(cs_emb)
            balanced_scores = np.zeros((n_queries, n_chunksets), dtype=np.float32)

            print(f"  Computing balanced scores for {n_queries} queries x {n_chunksets} chunksets...")

            for q_idx, q in enumerate(questions):
                # Get question embedding from pre-computed store
                if q.uid not in uid_to_q_idx:
                    print(f"  WARNING: No embedding for question {q.uid}")
                    continue
                q_vec = q_emb[uid_to_q_idx[q.uid]]
                q_norm_val = np_norm(q_vec)
                if q_norm_val < 1e-10:
                    continue  # Skip zero-norm queries

                # Compute chunkset similarities (suppress numerical warnings)
                cs_norms = np_norm(cs_emb, axis=1)
                valid_cs = cs_norms > 1e-10
                cs_sims = np.zeros(n_chunksets, dtype=np.float32)
                with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
                    cs_sims[valid_cs] = cs_emb[valid_cs] @ q_vec / (cs_norms[valid_cs] * q_norm_val)
                cs_sims = np.nan_to_num(cs_sims, nan=0.0, posinf=1.0, neginf=-1.0)

                # Compute chunk-level aggregated similarities
                agg_chunk_sims = np.zeros(n_chunksets, dtype=np.float32)
                for cs_idx, cs in enumerate(chunksets):
                    cs_key = f"{cs.doc_id}:chunkset:{cs.chunkset_id}"
                    if cs_key in chunk_map:
                        chunk_indices = chunk_map[cs_key]
                        if chunk_indices:
                            chunk_embs = chunk_emb[chunk_indices]
                            chunk_norms = np_norm(chunk_embs, axis=1)
                            valid = chunk_norms > 1e-10
                            if valid.any():
                                with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
                                    chunk_sims = chunk_embs[valid] @ q_vec / (chunk_norms[valid] * q_norm_val)
                                chunk_sims = np.nan_to_num(chunk_sims, nan=0.0, posinf=1.0, neginf=-1.0)
                                agg_chunk_sims[cs_idx] = agg_fn(chunk_sims)
                            else:
                                agg_chunk_sims[cs_idx] = cs_sims[cs_idx]
                        else:
                            agg_chunk_sims[cs_idx] = cs_sims[cs_idx]
                    else:
                        agg_chunk_sims[cs_idx] = cs_sims[cs_idx]

                # Optional per-query z-scoring before fusion
                if args.zscore:
                    # Z-score both signals for equal contribution
                    cs_mean, cs_std = cs_sims.mean(), cs_sims.std()
                    agg_mean, agg_std = agg_chunk_sims.mean(), agg_chunk_sims.std()

                    if cs_std > 1e-10:
                        cs_final = (cs_sims - cs_mean) / cs_std
                    else:
                        cs_final = cs_sims - cs_mean

                    if agg_std > 1e-10:
                        agg_final = (agg_chunk_sims - agg_mean) / agg_std
                    else:
                        agg_final = agg_chunk_sims - agg_mean
                else:
                    # Use raw scores (default)
                    cs_final = cs_sims
                    agg_final = agg_chunk_sims

                # Apply fusion
                w = args.weight
                if args.fusion == 'linear':
                    balanced_scores[q_idx] = fuse_linear(cs_final, agg_final, w)
                elif args.fusion == 'multiplicative':
                    for cs_idx in range(n_chunksets):
                        balanced_scores[q_idx, cs_idx] = fusion_fn(cs_final[cs_idx], agg_final[cs_idx], w)
                elif args.fusion == 'relu_boost':
                    for cs_idx in range(n_chunksets):
                        balanced_scores[q_idx, cs_idx] = fusion_fn(cs_final[cs_idx], agg_final[cs_idx], w)
                else:
                    # gated or other - need per-item fusion
                    for cs_idx in range(n_chunksets):
                        balanced_scores[q_idx, cs_idx] = fusion_fn(cs_final[cs_idx], agg_final[cs_idx], w)

            
            ranked_matrix = np.zeros((n_queries, min(EXTENDED_K, n_chunksets)), dtype=np.int32)
            for q_idx in range(n_queries):
                sorted_indices = np.argsort(-balanced_scores[q_idx])[:EXTENDED_K]
                ranked_matrix[q_idx] = sorted_indices
            
            def eval_fn(m, u, b, r):
                return evaluate_poma_at_budget(m, u, b, r, chunksets, raw_data)

        elif method == 'databricks_rcs_openai':
            from bench.vectors import load_vectors

            # Load pre-computed embeddings from .npz files
            try:
                txt_emb, txt_ids, _ = load_vectors("databricks_chunks")
                q_emb_all, q_ids, _ = load_vectors("questions")
            except FileNotFoundError as e:
                print(f"  ERROR: Vector file not found: {e}")
                print("  Run: python embed_all.py")
                continue

            # Build question UID to embedding index mapping
            uid_to_q_idx = {uid: idx for idx, uid in enumerate(q_ids)}

            # Get embeddings for just the questions we're evaluating
            q_emb = np.array([q_emb_all[uid_to_q_idx[q.uid]] for q in questions])

            txt_texts = [c.text for c in naive_chunks_txt]
            ranked_matrix = hybrid_rank_matrix(txt_emb, q_emb, txt_texts, q_texts, top_k=EXTENDED_K)

            def eval_fn(m, u, b, r):
                return evaluate_naive_at_budget(m, u, b, r, naive_chunks_txt, 'db_txt')

        elif method == 'unstructured_openai':
            from bench.vectors import load_vectors

            # Load pre-computed embeddings from .npz files
            try:
                us_emb, us_ids, _ = load_vectors("unstructured_elements")
                q_emb_all, q_ids, _ = load_vectors("questions")
            except FileNotFoundError as e:
                print(f"  ERROR: Vector file not found: {e}")
                print("  Run: python embed_all.py")
                continue

            # Build question UID to embedding index mapping
            uid_to_q_idx = {uid: idx for idx, uid in enumerate(q_ids)}

            # Get embeddings for just the questions we're evaluating
            q_emb = np.array([q_emb_all[uid_to_q_idx[q.uid]] for q in questions])

            us_texts = [c.text for c in unstructured_elements]
            ranked_matrix = hybrid_rank_matrix(us_emb, q_emb, us_texts, q_texts, top_k=EXTENDED_K)

            def eval_fn(m, u, b, r):
                return evaluate_naive_at_budget(m, u, b, r, unstructured_elements, 'us')

        else:
            print(f"Unknown method: {method}")
            continue

        run_adaptive_search(method, questions, ranked_matrix, eval_fn,
                           args.max_budget, args.initial_step, args.min_step)

    # Aggregate results
    print("\n" + "="*100)
    print("RESULTS (INDEX-BASED)")
    print("="*100)

    report_budgets = [5000, 10000, 25000, 50000, 100000, 250000, 500000, 1000000, 2000000]
    report_budgets = [b for b in report_budgets if b <= args.max_budget]

    all_results = {}
    for method in all_methods:
        all_results[method] = aggregate_results(method, uids, report_budgets)

    budget_headers = [f"@{b//1000}k" for b in report_budgets]
    print(f"\n{'Method':<25} {'Ans':>6} " + " ".join(f"{h:>6}" for h in budget_headers))
    print("-" * (32 + len(budget_headers) * 7))

    for method in all_methods:
        res = all_results[method]
        ans = f"{res['n_answered']}/{res['n_total']}"
        curve = res['accuracy_curve']
        accs = [f"{curve.get(b, 0)*100:5.1f}%" for b in report_budgets]
        print(f"{method:<25} {ans:>6} " + " ".join(accs))

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = script_dir / "results" / "bench_results" / f"bench_idx_{ts}.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump({
            'config': {
                'max_budget': args.max_budget,
                'evaluation': 'index_based',
            },
            'retrieval_params': retrieval_params,
            'results': all_results,
            'report_budgets': report_budgets,
            'timestamp': datetime.now().isoformat(),
        }, f, indent=2)
    print(f"\n[Results saved to {output_file}]")


if __name__ == "__main__":
    main()
