"""
Context building with cheatsheet deduplication.
"""

try:
    import tiktoken
except ImportError:
    tiktoken = None

try:
    from poma.retrieval import generate_cheatsheets
except ImportError:
    generate_cheatsheets = None

from .config import (
    NAIVE_CHUNK_SEPARATOR, CHEATSHEET_SEPARATOR, CHUNK_HEADER_TEMPLATE,
    MAX_BUDGET_OVERAGE, SKIP_CHEATSHEET_DEDUP,
)
from .models import ChunksetItem, NaiveChunk


# =====================================================================
# Token Counting
# =====================================================================

_tok_encoder = None

def n_tokens(text: str) -> int:
    """Count tokens in text using tiktoken (or fallback to char/4)."""
    global _tok_encoder
    if _tok_encoder is None:
        if tiktoken is None:
            return len(text) // 4
        _tok_encoder = tiktoken.get_encoding("cl100k_base")
    return len(_tok_encoder.encode(text or ""))


# =====================================================================
# Cheatsheet Context Building (POMA)
# =====================================================================

def build_cheatsheet_context(
    ranked_cs_indices: list[int],
    chunksets: list[ChunksetItem],
    raw_data: dict[str, tuple[list[dict], list[dict]]],
    budget_tokens: int,
) -> tuple[str, int, int, int]:
    """
    Build deduplicated context using generate_cheatsheets.

    Incrementally adds chunksets until the cheatsheet reaches the budget.
    Returns: (context_text, cheatsheet_tokens, raw_tokens_if_no_dedup, num_chunksets_used)
    """
    if generate_cheatsheets is None:
        raise RuntimeError("POMA mode requires generate_cheatsheets from poma.retrieval")

    by_doc: dict[str, list[dict]] = {}
    cs_used: list[int] = []
    raw_tokens = 0

    last_valid_context = ""
    last_valid_tokens = 0
    last_valid_raw_tokens = 0
    last_valid_cs_count = 0
    already_exceeded_once = False

    for cs_idx in ranked_cs_indices:
        cs = chunksets[cs_idx]
        doc_id = cs.doc_id

        if doc_id not in raw_data:
            continue

        raw_chunks, raw_chunksets = raw_data[doc_id]

        matching_raw_cs = None
        for rcs in raw_chunksets:
            if int(rcs.get("chunkset_index", -1)) == cs.chunkset_id:
                matching_raw_cs = rcs
                break

        if matching_raw_cs is None:
            continue

        if doc_id not in by_doc:
            by_doc[doc_id] = []
        by_doc[doc_id].append(matching_raw_cs)

        raw_tokens += n_tokens(cs.contents)
        cs_used.append(cs_idx)

        cheatsheet_parts = []

        for did, relevant_cs in by_doc.items():
            if not relevant_cs:
                continue
            rchunks, _ = raw_data.get(did, ([], []))
            if SKIP_CHEATSHEET_DEDUP:
                for rcs in relevant_cs:
                    text = str(rcs.get("contents", ""))
                    cs_id = rcs.get("chunkset_index", "?")
                    header = CHUNK_HEADER_TEMPLATE.format(source_id=f"doc{did}_cs{cs_id}")
                    cheatsheet_parts.append(header + text)
            else:
                try:
                    sheets = generate_cheatsheets(relevant_cs, rchunks)
                    for sheet_idx, sheet in enumerate(sheets):
                        sheet_text = str(sheet.get("cheatsheet", "") or sheet.get("content", ""))
                        header = CHUNK_HEADER_TEMPLATE.format(source_id=f"doc{did}_cs{sheet_idx}")
                        cheatsheet_parts.append(header + sheet_text)
                except Exception as e:
                    import logging
                    logging.warning(f"Cheatsheet generation failed for doc {did}: {e}")
                    for rcs in relevant_cs:
                        text = str(rcs.get("contents", ""))
                        cs_id = rcs.get("chunkset_index", "?")
                        header = CHUNK_HEADER_TEMPLATE.format(source_id=f"doc{did}_cs{cs_id}")
                        cheatsheet_parts.append(header + text)

        context = CHEATSHEET_SEPARATOR.join(cheatsheet_parts)
        total_tokens = n_tokens(context)

        if total_tokens > budget_tokens:
            overage = total_tokens - budget_tokens
            if already_exceeded_once or overage > MAX_BUDGET_OVERAGE:
                by_doc[doc_id].pop()
                if not by_doc[doc_id]:
                    del by_doc[doc_id]
                cs_used.pop()
                raw_tokens -= n_tokens(cs.contents)
                return last_valid_context, last_valid_tokens, last_valid_raw_tokens, last_valid_cs_count
            else:
                already_exceeded_once = True

        last_valid_context = context
        last_valid_tokens = total_tokens
        last_valid_raw_tokens = raw_tokens
        last_valid_cs_count = len(cs_used)

        if already_exceeded_once:
            break

    return last_valid_context, last_valid_tokens, last_valid_raw_tokens, last_valid_cs_count


# =====================================================================
# Fast Raw Chunkset Context (for probing - no cheatsheet generation)
# =====================================================================

def build_raw_chunkset_context(
    ranked_cs_indices: list[int],
    chunksets: list[ChunksetItem],
    budget_tokens: int,
) -> tuple[str, int, list[int]]:
    """
    Build context from raw chunkset contents (no cheatsheet generation).
    This is O(n) and used for fast evidence probing.

    Returns: (context_text, tokens_used, chunkset_indices_used)
    """
    context_parts = []
    total_tokens = 0
    indices_used = []
    separator_toks = n_tokens(CHEATSHEET_SEPARATOR)
    already_exceeded_once = False

    for cs_idx in ranked_cs_indices:
        cs = chunksets[cs_idx]
        header = CHUNK_HEADER_TEMPLATE.format(source_id=f"doc{cs.doc_id}_cs{cs.chunkset_id}")
        text = header + cs.contents
        toks = n_tokens(text)

        sep_overhead = separator_toks if indices_used else 0
        projected_total = total_tokens + toks + sep_overhead

        if projected_total > budget_tokens:
            overage = projected_total - budget_tokens
            if already_exceeded_once or overage > MAX_BUDGET_OVERAGE:
                break
            else:
                already_exceeded_once = True

        context_parts.append(text)
        total_tokens = projected_total
        indices_used.append(cs_idx)

        if already_exceeded_once:
            break

    context = CHEATSHEET_SEPARATOR.join(context_parts)
    return context, n_tokens(context), indices_used


def build_cheatsheet_from_indices(
    cs_indices: list[int],
    chunksets: list[ChunksetItem],
    raw_data: dict[str, tuple[list[dict], list[dict]]],
) -> tuple[str, int]:
    """
    Build cheatsheet from specific chunkset indices (single pass, no iteration).
    Used after raw probing finds evidence - we build the actual cheatsheet ONCE.

    Returns: (cheatsheet_context, token_count)
    """
    if generate_cheatsheets is None:
        raise RuntimeError("POMA mode requires generate_cheatsheets from poma.retrieval")

    # Group by document
    by_doc: dict[str, list[dict]] = {}

    for cs_idx in cs_indices:
        cs = chunksets[cs_idx]
        doc_id = cs.doc_id

        if doc_id not in raw_data:
            continue

        raw_chunks, raw_chunksets = raw_data[doc_id]

        matching_raw_cs = None
        for rcs in raw_chunksets:
            if int(rcs.get("chunkset_index", -1)) == cs.chunkset_id:
                matching_raw_cs = rcs
                break

        if matching_raw_cs is None:
            continue

        if doc_id not in by_doc:
            by_doc[doc_id] = []
        by_doc[doc_id].append(matching_raw_cs)

    # Generate cheatsheets once per document
    cheatsheet_parts = []

    for did, relevant_cs in by_doc.items():
        if not relevant_cs:
            continue
        rchunks, _ = raw_data.get(did, ([], []))

        if SKIP_CHEATSHEET_DEDUP:
            for rcs in relevant_cs:
                text = str(rcs.get("contents", ""))
                cs_id = rcs.get("chunkset_index", "?")
                header = CHUNK_HEADER_TEMPLATE.format(source_id=f"doc{did}_cs{cs_id}")
                cheatsheet_parts.append(header + text)
        else:
            try:
                sheets = generate_cheatsheets(relevant_cs, rchunks)
                for sheet_idx, sheet in enumerate(sheets):
                    sheet_text = str(sheet.get("cheatsheet", "") or sheet.get("content", ""))
                    header = CHUNK_HEADER_TEMPLATE.format(source_id=f"doc{did}_cs{sheet_idx}")
                    cheatsheet_parts.append(header + sheet_text)
            except Exception as e:
                import logging
                logging.warning(f"Cheatsheet generation failed for doc {did}: {e}")
                for rcs in relevant_cs:
                    text = str(rcs.get("contents", ""))
                    cs_id = rcs.get("chunkset_index", "?")
                    header = CHUNK_HEADER_TEMPLATE.format(source_id=f"doc{did}_cs{cs_id}")
                    cheatsheet_parts.append(header + text)

    context = CHEATSHEET_SEPARATOR.join(cheatsheet_parts)
    return context, n_tokens(context)


# =====================================================================
# Naive Chunk Context Building
# =====================================================================

def build_naive_chunk_context(
    ranked_indices: list[int],
    naive_chunks: list[NaiveChunk],
    budget_tokens: int,
) -> tuple[str, int, int]:
    """Build context from naive chunks up to token budget.

    Returns: (context_text, tokens_used, num_chunks_used)
    """
    context_parts = []
    total_tokens = 0
    num_used = 0
    separator_toks = n_tokens(NAIVE_CHUNK_SEPARATOR)
    already_exceeded_once = False

    for idx in ranked_indices:
        chunk = naive_chunks[idx]
        header = CHUNK_HEADER_TEMPLATE.format(source_id=f"doc{chunk.doc_id}_chunk{chunk.chunk_id}")
        text = header + chunk.text
        toks = n_tokens(text)

        sep_overhead = separator_toks if num_used > 0 else 0
        projected_total = total_tokens + toks + sep_overhead

        if projected_total > budget_tokens:
            overage = projected_total - budget_tokens
            if already_exceeded_once or overage > MAX_BUDGET_OVERAGE:
                break
            else:
                already_exceeded_once = True

        context_parts.append(text)
        total_tokens = projected_total
        num_used += 1

        if already_exceeded_once:
            break

    context = NAIVE_CHUNK_SEPARATOR.join(context_parts)
    return context, n_tokens(context), num_used
