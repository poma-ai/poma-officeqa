"""
Evaluation by INDEX presence: check if golden chunk indices are retrieved.

This is the benchmark the user wants:
- For each mode (poma, db_txt, us), retrieve chunks sorted by relevance
- Check if ALL indices from found_in.{mode} are present in retrieved set
- Report min tokens needed to include all golden indices
"""

import json
import numpy as np
from pathlib import Path
from typing import Any

from .config import EVIDENCE_FILE, MAX_BUDGET_OVERAGE
from .models import ChunksetItem, NaiveChunk
from .context import n_tokens, build_cheatsheet_context, build_naive_chunk_context


# =====================================================================
# Load evidence with found_in indices
# =====================================================================

_evidence_cache: dict = {}

# Global offset maps (built once per run): {mode: {doc_name: global_offset}}
_offset_maps: dict[str, dict[str, int]] = {}


def set_corpus_offsets(mode: str, doc_offsets: dict[str, int]):
    """Set document offset map for a corpus mode (called by benchmark at startup)."""
    global _offset_maps
    _offset_maps[mode] = doc_offsets


def _load_evidence():
    """Load evidence_requirements.json (final version with verified indices)."""
    global _evidence_cache
    # Try final version first, then fall back to regular
    evidence_path = Path(EVIDENCE_FILE).parent / "evidence_requirements_final.json"
    if not evidence_path.exists():
        evidence_path = EVIDENCE_FILE

    if evidence_path.exists():
        with open(evidence_path) as f:
            data = json.load(f)
            _evidence_cache = data.get('UIDs', {})
    return bool(_evidence_cache)


def get_golden_indices(uid: str, mode: str) -> list[int]:
    """
    Get golden chunk indices for a question and mode.
    
    Evidence file stores (doc, local_id) tuples. This function converts
    them to GLOBAL indices using the offset map for benchmark evaluation.

    Args:
        uid: Question UID (e.g., 'UID0001')
        mode: 'poma', 'db_txt', or 'us'

    Returns:
        List of global chunk indices that must be retrieved
    """
    if not _evidence_cache:
        _load_evidence()

    uid_data = _evidence_cache.get(uid, {})
    docs = uid_data.get('documents', [])

    if not docs:
        return []

    found_in = docs[0].get('found_in', {})
    entries = found_in.get(mode, [])
    
    if not entries:
        return []
    
    global_indices = []
    for entry in entries:
        # New format: {"doc": "...", "local_id": N}
        if isinstance(entry, dict):
            doc_name = entry.get('doc', '')
            local_id = entry.get('local_id', 0)
            if mode in _offset_maps and doc_name in _offset_maps[mode]:
                offset = _offset_maps[mode][doc_name]
                global_indices.append(offset + local_id)
            else:
                # Offset not set - can't convert
                pass
        else:
            # Old format: bare integer (backward compat, assume global)
            global_indices.append(entry)
    
    return global_indices


def get_valid_uids() -> set[str]:
    """Get all valid UIDs from evidence_requirements.json."""
    if not _evidence_cache:
        _load_evidence()
    return set(_evidence_cache.keys())


def get_valid_uids_for_mode(mode: str) -> list[str]:
    """Get UIDs that have golden indices for the specified mode."""
    if not _evidence_cache:
        _load_evidence()

    valid = []
    for uid in _evidence_cache:
        indices = get_golden_indices(uid, mode)
        if indices:
            valid.append(uid)
    return valid


# =====================================================================
# POMA Evaluation by Index
# =====================================================================

def evaluate_poma_by_index(
    questions: list,
    ranked_matrix: np.ndarray,
    chunksets: list[ChunksetItem],
    raw_data: dict,
    max_budget: int = 100000,
    budget_step: int = 1000,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Evaluate POMA retrieval by checking if golden indices are retrieved.

    For each question:
    1. Get golden indices from evidence_requirements.found_in.poma
    2. Increase budget until ALL golden indices are in retrieved set
    3. Report cheatsheet (deduplicated) token count at that point
    """
    min_budgets = []
    question_details = {}

    # Build global index to chunkset mapping
    all_chunksets = chunksets

    print(f"Evaluating {len(questions)} questions (INDEX-based, POMA, max {max_budget:,} tokens)...")

    for qi, q in enumerate(questions):
        ranked = ranked_matrix[qi, :].tolist()  # Global indices, sorted by relevance

        # Get golden indices for this question
        golden_indices = set(get_golden_indices(q.uid, 'poma'))

        if not golden_indices:
            min_budgets.append(float('inf'))
            question_details[q.uid] = {
                "passed": False,
                "min_budget": None,
                "reason": "no_golden_indices_for_poma",
                "golden_count": 0,
                "found_count": 0,
            }
            if verbose:
                print(f"  [{q.uid}] ✗ No golden indices for POMA")
            continue

        # Find minimum budget that retrieves ALL golden indices
        found = False
        min_budget = None
        final_tokens = 0
        n_found = 0

        for budget in range(budget_step, max_budget + 1, budget_step):
            # Build context and track which chunksets were included
            context, toks, raw_toks, n_cs = build_cheatsheet_context(
                ranked, all_chunksets, raw_data, budget
            )

            if not context:
                continue

            # Get set of retrieved indices at this budget
            # We need to figure out which chunksets were actually included
            # This requires tracking during context building, but for now
            # we can approximate by taking top N from ranked list
            retrieved_indices = set()
            current_tokens = 0
            for idx in ranked:
                if idx < 0 or idx >= len(all_chunksets):
                    continue
                cs = all_chunksets[idx]
                cs_tokens = cs.n_tokens
                if current_tokens + cs_tokens > budget + MAX_BUDGET_OVERAGE:
                    break
                retrieved_indices.add(idx)
                current_tokens += cs_tokens

            # Check coverage
            n_found = len(golden_indices & retrieved_indices)

            if golden_indices <= retrieved_indices:  # All golden indices retrieved
                found = True
                min_budget = toks  # Use cheatsheet tokens (after dedup)
                final_tokens = toks
                break

        if found:
            min_budgets.append(min_budget)
            question_details[q.uid] = {
                "passed": True,
                "min_budget": min_budget,
                "reason": f"indices:{len(golden_indices)}/{len(golden_indices)}@{final_tokens}tok",
                "golden_count": len(golden_indices),
                "found_count": len(golden_indices),
            }
        else:
            min_budgets.append(float('inf'))
            question_details[q.uid] = {
                "passed": False,
                "min_budget": None,
                "reason": f"indices_missing:{n_found}/{len(golden_indices)}",
                "golden_count": len(golden_indices),
                "found_count": n_found,
            }

        if verbose:
            status = "✓" if found else "✗"
            budget_str = f"{min_budget:,}" if found else "NOT FOUND"
            print(f"  [{q.uid}] {status} @ {budget_str} tokens | {question_details[q.uid]['reason']}")

    # Compute statistics
    n_total = len(questions)
    n_answered = sum(1 for d in question_details.values() if d["passed"])
    found_budgets = [mb for mb in min_budgets if mb != float('inf')]
    avg_min_budget = float(np.mean(found_budgets)) if found_budgets else 0.0
    median_min_budget = float(np.median(found_budgets)) if found_budgets else 0.0

    # Accuracy curve
    accuracy_curve = {}
    for budget in range(budget_step, max_budget + 1, budget_step):
        n_at_budget = sum(1 for mb in min_budgets if mb != float('inf') and mb <= budget + MAX_BUDGET_OVERAGE)
        accuracy_curve[budget] = n_at_budget / n_total if n_total > 0 else 0.0

    return {
        "mode": "poma",
        "evaluation_method": "index_presence",
        "avg_min_budget": avg_min_budget,
        "median_min_budget": median_min_budget,
        "min_budgets": [mb if mb != float('inf') else None for mb in min_budgets],
        "accuracy_curve": accuracy_curve,
        "n_answered": n_answered,
        "n_total": n_total,
        "questions": question_details,
    }


# =====================================================================
# Naive (DB_TXT / US) Evaluation by Index
# =====================================================================

def evaluate_naive_by_index(
    questions: list,
    ranked_matrix: np.ndarray,
    naive_chunks: list[NaiveChunk],
    mode: str,  # 'db_txt' or 'us'
    max_budget: int = 100000,
    budget_step: int = 1000,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Evaluate naive chunk retrieval by checking if golden indices are retrieved.

    For each question:
    1. Get golden indices from evidence_requirements.found_in.{mode}
    2. Increase budget until ALL golden indices are in retrieved set
    3. Report token count at that point
    """
    min_budgets = []
    question_details = {}

    print(f"Evaluating {len(questions)} questions (INDEX-based, {mode}, max {max_budget:,} tokens)...")

    for qi, q in enumerate(questions):
        ranked = ranked_matrix[qi, :].tolist()  # Global indices, sorted by relevance

        # Get golden indices for this question
        golden_indices = set(get_golden_indices(q.uid, mode))

        if not golden_indices:
            min_budgets.append(float('inf'))
            question_details[q.uid] = {
                "passed": False,
                "min_budget": None,
                "reason": f"no_golden_indices_for_{mode}",
                "golden_count": 0,
                "found_count": 0,
            }
            if verbose:
                print(f"  [{q.uid}] ✗ No golden indices for {mode}")
            continue

        # Find minimum budget that retrieves ALL golden indices
        found = False
        min_budget = None
        final_tokens = 0
        n_found = 0

        for budget in range(budget_step, max_budget + 1, budget_step):
            # Determine which chunks would be retrieved at this budget
            retrieved_indices = set()
            current_tokens = 0

            for idx in ranked:
                if idx < 0 or idx >= len(naive_chunks):
                    continue
                chunk = naive_chunks[idx]
                chunk_tokens = n_tokens(chunk.text)
                if current_tokens + chunk_tokens > budget + MAX_BUDGET_OVERAGE:
                    break
                retrieved_indices.add(idx)
                current_tokens += chunk_tokens

            # Check coverage
            n_found = len(golden_indices & retrieved_indices)

            if golden_indices <= retrieved_indices:  # All golden indices retrieved
                found = True
                min_budget = current_tokens
                final_tokens = current_tokens
                break

        if found:
            min_budgets.append(min_budget)
            question_details[q.uid] = {
                "passed": True,
                "min_budget": min_budget,
                "reason": f"indices:{len(golden_indices)}/{len(golden_indices)}@{final_tokens}tok",
                "golden_count": len(golden_indices),
                "found_count": len(golden_indices),
            }
        else:
            min_budgets.append(float('inf'))
            question_details[q.uid] = {
                "passed": False,
                "min_budget": None,
                "reason": f"indices_missing:{n_found}/{len(golden_indices)}",
                "golden_count": len(golden_indices),
                "found_count": n_found,
            }

        if verbose:
            status = "✓" if found else "✗"
            budget_str = f"{min_budget:,}" if found else "NOT FOUND"
            print(f"  [{q.uid}] {status} @ {budget_str} tokens | {question_details[q.uid]['reason']}")

    # Compute statistics
    n_total = len(questions)
    n_answered = sum(1 for d in question_details.values() if d["passed"])
    found_budgets = [mb for mb in min_budgets if mb != float('inf')]
    avg_min_budget = float(np.mean(found_budgets)) if found_budgets else 0.0
    median_min_budget = float(np.median(found_budgets)) if found_budgets else 0.0

    # Accuracy curve
    accuracy_curve = {}
    for budget in range(budget_step, max_budget + 1, budget_step):
        n_at_budget = sum(1 for mb in min_budgets if mb != float('inf') and mb <= budget + MAX_BUDGET_OVERAGE)
        accuracy_curve[budget] = n_at_budget / n_total if n_total > 0 else 0.0

    return {
        "mode": mode,
        "evaluation_method": "index_presence",
        "avg_min_budget": avg_min_budget,
        "median_min_budget": median_min_budget,
        "min_budgets": [mb if mb != float('inf') else None for mb in min_budgets],
        "accuracy_curve": accuracy_curve,
        "n_answered": n_answered,
        "n_total": n_total,
        "questions": question_details,
    }
