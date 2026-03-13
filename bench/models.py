"""
Data models for benchmark: NaiveChunk, ChunksetItem, OfficeQAQuestion, EmbeddingResult.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class NaiveChunk:
    """A chunk from naive (non-POMA) chunking."""
    doc_id: str
    chunk_id: int
    text: str


@dataclass
class ChunksetItem:
    """A chunkset from POMA processing."""
    doc_id: str
    chunkset_id: int
    contents: str
    to_embed: str
    path_texts: list[str]
    path_indices: list[int]
    path_depths: list[int]
    leaf_idx: Optional[int] = None
    canonical_ancestors: Optional[list[int]] = None
    parent_map: Optional[dict[int, Optional[int]]] = None


@dataclass
class OfficeQAQuestion:
    """A question from the OfficeQA dataset."""
    uid: str
    question: str
    answer: str
    source_file: str
    difficulty: str
    doc_id: Optional[str] = None


@dataclass
class EmbeddingResult:
    """Result of chunksets-first embedding for Voyage.
    
    MaxSim uses: max(prefix_vec @ query, leaf_vec @ query) per chunkset.
    """
    chunkset_leaf_vecs: np.ndarray      # (n_cs, dim) - leaf content vectors
    chunkset_prefix_vecs: np.ndarray    # (n_cs, dim) - prefix context vectors
    chunkset_mats: list[np.ndarray]     # Legacy: [prefix, leaf] stacked per chunkset
    chunk_vecs: dict[tuple[str, int], np.ndarray]  # Not used in 2-tuple mode
    chunk_to_chunksets: dict[tuple[str, int], list[int]]  # Not used in 2-tuple mode
