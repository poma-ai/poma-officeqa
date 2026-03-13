"""
Bench package: Context recall benchmark for POMA chunking.
"""

from .config import (
    # Constants
    OPENAI_EMBED_MODEL, OPENAI_EMBED_DIMS,
    TOKEN_BUDGETS, NAIVE_CHUNK_SIZE, NAIVE_CHUNK_OVERLAP,
    EXTENDED_K, SKIP_CHEATSHEET_DEDUP,
    VECTORS_DIR, RESULTS_DIR,
    TRANSFORMED_DIR, UNSTRUCTURED_DIR, EVIDENCE_FILE,
    # Classes
    RunConfig, BenchmarkRun, ResultsStore,
    # Functions
    create_run_id, print_results_comparison,
)

from .models import (
    NaiveChunk, ChunksetItem, OfficeQAQuestion, EmbeddingResult,
)

from .embeddings import (
    norm,
)

from .retrieval import (
    hybrid_rank_matrix, retrieve_by_maxsim,
)

from .context import (
    n_tokens,
    build_cheatsheet_context, build_naive_chunk_context,
    build_raw_chunkset_context, build_cheatsheet_from_indices,
)

from .loaders import (
    list_poma_files, get_poma_filename, load_chunksets_from_poma,
    load_poma_corpus, load_raw_poma_data,
    load_naive_chunks_from_txt, load_unstructured_elements,
    load_officeqa,
)

from .evaluation_index import (
    get_valid_uids, get_golden_indices, get_valid_uids_for_mode,
    set_corpus_offsets,
    evaluate_poma_by_index, evaluate_naive_by_index,
)

from .aggregation import (
    # Chunk aggregation methods
    agg_max, agg_topk_mean, agg_percentile, agg_softmax_pool,
    agg_log_mean_exp, agg_spike_score, agg_max_adjusted, agg_smoothmax_adjusted,
    # Score fusion
    fuse_linear,
    # Registries
    AGGREGATION_METHODS, FUSION_METHODS,
)

__all__ = [
    # Config
    'OPENAI_EMBED_MODEL', 'OPENAI_EMBED_DIMS',
    'TOKEN_BUDGETS', 'EXTENDED_K', 'SKIP_CHEATSHEET_DEDUP',
    'VECTORS_DIR', 'RESULTS_DIR',
    'RunConfig', 'BenchmarkRun', 'ResultsStore',
    'create_run_id', 'print_results_comparison',
    # Models
    'NaiveChunk', 'ChunksetItem', 'OfficeQAQuestion', 'EmbeddingResult',
    # Embeddings
    'norm',
    # Retrieval
    'hybrid_rank_matrix', 'retrieve_by_maxsim',
    # Context
    'n_tokens',
    'build_cheatsheet_context', 'build_naive_chunk_context',
    # Loaders
    'list_poma_files', 'get_poma_filename', 'load_chunksets_from_poma',
    'load_poma_corpus', 'load_raw_poma_data',
    'load_naive_chunks_from_txt', 'load_unstructured_elements',
    'load_officeqa',
    # Evaluation
    'get_valid_uids', 'get_golden_indices', 'get_valid_uids_for_mode',
    'set_corpus_offsets',
    'evaluate_poma_by_index', 'evaluate_naive_by_index',
    # Aggregation
    'agg_max', 'agg_max_adjusted', 'fuse_linear',
    'AGGREGATION_METHODS', 'FUSION_METHODS',
]
