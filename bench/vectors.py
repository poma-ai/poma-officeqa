"""
Vector store for embeddings - clean, Qdrant-compatible design.

Structure:
    vectors/
    ├── questions.npz           # Question embeddings
    ├── poma_chunksets.npz      # POMA chunkset embeddings
    ├── poma_chunks.npz         # POMA chunk embeddings (for balanced scoring)
    ├── databricks_chunks.npz   # Databricks naive chunk embeddings
    └── unstructured_elements.npz  # Unstructured element embeddings

Each .npz file contains:
    - embeddings: np.ndarray, shape (N, dims)
    - ids: np.ndarray of strings, e.g. "treasury_bulletin_1941_01:chunkset:188"
    - metadata: JSON string with model, dims, created, count

Qdrant migration:
    data = np.load("poma_chunksets.npz", allow_pickle=True)
    qdrant.upload_collection(
        collection_name="poma_chunksets",
        vectors=data["embeddings"],
        ids=list(range(len(data["ids"]))),
        payload=[{"id": id} for id in data["ids"]]
    )
"""

import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional

from openai import OpenAI

# Constants
EMBED_MODEL = "text-embedding-3-large"
EMBED_DIMS = 1536
VECTORS_DIR = Path(__file__).parent.parent / "vectors"


def _get_client() -> OpenAI:
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def _normalize(v: np.ndarray) -> np.ndarray:
    """L2 normalize vectors."""
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return v / norms


def embed_texts(texts: list[str], batch_size: int = 100, show_progress: bool = True) -> np.ndarray:
    """Embed texts using OpenAI API. Returns normalized vectors."""
    client = _get_client()
    all_embeddings = []

    # Handle empty texts
    safe_texts = [t if t and t.strip() else "[empty]" for t in texts]

    n_batches = (len(safe_texts) + batch_size - 1) // batch_size
    for i in range(0, len(safe_texts), batch_size):
        batch = safe_texts[i:i + batch_size]
        if show_progress:
            batch_num = i // batch_size + 1
            print(f"  Embedding batch {batch_num}/{n_batches} ({len(batch)} texts)...")

        response = client.embeddings.create(
            model=EMBED_MODEL,
            input=batch,
            dimensions=EMBED_DIMS
        )
        embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(embeddings)

    result = np.array(all_embeddings, dtype=np.float32)
    return _normalize(result)


def save_vectors(name: str, ids: list[str], embeddings: np.ndarray, extra_metadata: dict = None):
    """Save vectors to .npz file with metadata."""
    VECTORS_DIR.mkdir(parents=True, exist_ok=True)

    metadata = {
        "model": EMBED_MODEL,
        "dims": EMBED_DIMS,
        "count": len(ids),
        "created": datetime.now().isoformat(),
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    filepath = VECTORS_DIR / f"{name}.npz"
    np.savez(
        filepath,
        embeddings=embeddings.astype(np.float32),
        ids=np.array(ids, dtype=object),
        metadata=json.dumps(metadata)
    )

    size_mb = filepath.stat().st_size / (1024 * 1024)
    print(f"  Saved {filepath.name}: {len(ids):,} vectors, {size_mb:.1f} MB")
    return filepath


def load_vectors(name: str) -> tuple[np.ndarray, list[str], dict]:
    """Load vectors from .npz file(s). Returns (embeddings, ids, metadata).

    Supports both single-file format (name.npz) and split-directory format
    (name/doc_*.npz) for files that exceed GitHub's 100MB limit.
    """
    # Check for single file first
    single_file = VECTORS_DIR / f"{name}.npz"
    if single_file.exists():
        data = np.load(single_file, allow_pickle=True)
        embeddings = data["embeddings"]
        ids = list(data["ids"])
        metadata = json.loads(str(data["metadata"]))
        return embeddings, ids, metadata

    # Check for split directory
    split_dir = VECTORS_DIR / name
    if split_dir.is_dir():
        # Support both old format (doc_*.npz) and new hash format (treasury_bulletin_*_hash.npz)
        doc_files = list(split_dir.glob("doc_*.npz"))
        if doc_files:
            doc_files = sorted(doc_files, key=lambda x: int(x.stem.split('_')[1]))
        else:
            # New format: treasury_bulletin_{year}_{month}_{hash}.npz
            doc_files = sorted(split_dir.glob("treasury_bulletin_*.npz"))
        if not doc_files:
            raise FileNotFoundError(f"No vector files found in {split_dir}")

        all_embeddings = []
        all_ids = []
        merged_metadata = None
        merged_chunk_map = {}  # For poma_chunks

        # Track index offset for chunk_map remapping
        index_offset = 0

        for doc_file in doc_files:
            data = np.load(doc_file, allow_pickle=True)
            doc_embeddings = data["embeddings"]
            doc_ids = list(data["ids"])

            all_embeddings.append(doc_embeddings)
            all_ids.extend(doc_ids)

            # Handle metadata if present (old format has it, new format may not)
            if "metadata" in data.files:
                doc_metadata = json.loads(str(data["metadata"]))
                # Merge chunk_map with index offset
                if "chunk_map" in doc_metadata:
                    doc_chunk_map = json.loads(doc_metadata["chunk_map"])
                    for key, indices in doc_chunk_map.items():
                        merged_chunk_map[key] = [i + index_offset for i in indices]
                # Keep first file's metadata as base
                if merged_metadata is None:
                    merged_metadata = doc_metadata.copy()

            index_offset += len(doc_ids)

        # Update merged metadata
        if merged_metadata is None:
            merged_metadata = {"model": "text-embedding-3-large", "dims": 1536}
        merged_metadata["count"] = len(all_ids)
        merged_metadata.pop("doc_id", None)
        merged_metadata.pop("part", None)
        merged_metadata.pop("total_parts", None)
        if merged_chunk_map:
            merged_metadata["chunk_map"] = json.dumps(merged_chunk_map)

        embeddings = np.vstack(all_embeddings)
        return embeddings, all_ids, merged_metadata

    raise FileNotFoundError(f"Vector file not found: {single_file} or {split_dir}/")


def vectors_exist(name: str) -> bool:
    """Check if vector file exists (single file or split directory)."""
    single_file = VECTORS_DIR / f"{name}.npz"
    split_dir = VECTORS_DIR / name
    if single_file.exists():
        return True
    if split_dir.is_dir():
        # Support both old and new formats
        return any(split_dir.glob("doc_*.npz")) or any(split_dir.glob("treasury_bulletin_*.npz"))
    return False


def load_prefix_leaf_vectors(name: str = "poma_prefix_leaf") -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load prefix/leaf vectors. Returns (prefix_emb, leaf_emb, ids)."""
    filepath = VECTORS_DIR / f"{name}.npz"
    if not filepath.exists():
        raise FileNotFoundError(f"Prefix/leaf vectors not found: {filepath}")

    data = np.load(filepath, allow_pickle=True)
    prefix_emb = data["prefix_embeddings"]
    leaf_emb = data["leaf_embeddings"]
    ids = list(data["ids"])
    return prefix_emb, leaf_emb, ids


def list_vector_files() -> list[dict]:
    """List all vector files with their metadata."""
    files = []

    # Single .npz files
    for f in VECTORS_DIR.glob("*.npz"):
        try:
            data = np.load(f, allow_pickle=True)
            metadata = json.loads(str(data["metadata"]))
            metadata["file"] = f.name
            metadata["size_mb"] = f.stat().st_size / (1024 * 1024)
            files.append(metadata)
        except Exception as e:
            files.append({"file": f.name, "error": str(e)})

    # Split directories
    for d in VECTORS_DIR.iterdir():
        if d.is_dir():
            doc_files = list(d.glob("doc_*.npz"))
            if doc_files:
                try:
                    # Sum up sizes and counts
                    total_size = sum(f.stat().st_size for f in doc_files)
                    total_count = 0
                    sample_metadata = None
                    for f in doc_files:
                        data = np.load(f, allow_pickle=True)
                        meta = json.loads(str(data["metadata"]))
                        total_count += meta.get("count", 0)
                        if sample_metadata is None:
                            sample_metadata = meta
                    if sample_metadata:
                        sample_metadata["file"] = f"{d.name}/ ({len(doc_files)} files)"
                        sample_metadata["size_mb"] = total_size / (1024 * 1024)
                        sample_metadata["count"] = total_count
                        files.append(sample_metadata)
                except Exception as e:
                    files.append({"file": d.name, "error": str(e)})

    return files


# ============================================================
# High-level embedding functions for each corpus type
# ============================================================

def embed_questions(questions: list, force: bool = False) -> np.ndarray:
    """Embed question texts. Returns embeddings array."""
    name = "questions"

    if vectors_exist(name) and not force:
        embeddings, ids, _ = load_vectors(name)
        print(f"  Loaded {name}: {len(ids)} vectors")
        return embeddings

    print(f"Embedding {len(questions)} questions...")
    texts = [q.question for q in questions]
    ids = [q.uid for q in questions]
    embeddings = embed_texts(texts)
    save_vectors(name, ids, embeddings)
    return embeddings


def embed_poma_chunksets(chunksets: list, force: bool = False) -> np.ndarray:
    """Embed POMA chunkset contents using pre-computed to_embed field."""
    name = "poma_chunksets"

    if vectors_exist(name) and not force:
        embeddings, ids, _ = load_vectors(name)
        print(f"  Loaded {name}: {len(ids)} vectors")
        return embeddings

    print(f"Embedding {len(chunksets)} POMA chunksets...")
    texts = [cs.to_embed for cs in chunksets]
    ids = [f"{cs.doc_id}:chunkset:{cs.chunkset_id}" for cs in chunksets]
    embeddings = embed_texts(texts)
    save_vectors(name, ids, embeddings)
    return embeddings


def embed_poma_prefix_leaf(chunksets: list, force: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Embed POMA chunksets with separate prefix and leaf vectors for MaxSim retrieval.

    MaxSim computes: max(prefix @ query, leaf @ query)
    This lets structural context (prefix) and data values (leaf) match independently.

    Returns:
        prefix_embeddings: (n_chunksets, dims) - structural context vectors
        leaf_embeddings: (n_chunksets, dims) - data value vectors
    """
    name = "poma_prefix_leaf"

    if vectors_exist(name) and not force:
        data = np.load(VECTORS_DIR / f"{name}.npz", allow_pickle=True)
        prefix_emb = data["prefix_embeddings"]
        leaf_emb = data["leaf_embeddings"]
        print(f"  Loaded {name}: {len(prefix_emb)} prefix/leaf pairs")
        return prefix_emb, leaf_emb

    print(f"Embedding {len(chunksets)} POMA chunksets as prefix/leaf pairs...")

    prefix_texts = []
    leaf_texts = []
    ids = []

    for cs in chunksets:
        cs_id = f"{cs.doc_id}:chunkset:{cs.chunkset_id}"
        ids.append(cs_id)

        # Get leaf index - default to last element if not specified
        leaf_idx = cs.leaf_idx
        if leaf_idx is None:
            leaf_idx = len(cs.path_texts) - 1 if cs.path_texts else 0

        if cs.path_texts and len(cs.path_texts) > 0:
            # Prefix: everything before the leaf (structural context)
            if leaf_idx > 0:
                prefix = "\n".join(cs.path_texts[:leaf_idx])
            else:
                # No prefix - use first chunk as both
                prefix = cs.path_texts[0] if cs.path_texts else ""

            # Leaf: the actual data content
            if leaf_idx < len(cs.path_texts):
                leaf = cs.path_texts[leaf_idx]
            else:
                leaf = cs.path_texts[-1] if cs.path_texts else ""
        else:
            # Fallback to full contents for both
            prefix = cs.contents
            leaf = cs.contents

        prefix_texts.append(prefix)
        leaf_texts.append(leaf)

    # Embed prefix and leaf texts separately
    print(f"  Embedding {len(prefix_texts)} prefix texts...")
    prefix_embeddings = embed_texts(prefix_texts, show_progress=True)

    print(f"  Embedding {len(leaf_texts)} leaf texts...")
    leaf_embeddings = embed_texts(leaf_texts, show_progress=True)

    # Save both in one file
    VECTORS_DIR.mkdir(parents=True, exist_ok=True)
    filepath = VECTORS_DIR / f"{name}.npz"

    metadata = {
        "model": EMBED_MODEL,
        "dims": EMBED_DIMS,
        "count": len(ids),
        "created": datetime.now().isoformat(),
        "description": "Prefix (structural context) and leaf (data values) embeddings for MaxSim"
    }

    np.savez(
        filepath,
        prefix_embeddings=prefix_embeddings.astype(np.float32),
        leaf_embeddings=leaf_embeddings.astype(np.float32),
        ids=np.array(ids, dtype=object),
        metadata=json.dumps(metadata)
    )

    size_mb = filepath.stat().st_size / (1024 * 1024)
    print(f"  Saved {filepath.name}: {len(ids):,} prefix/leaf pairs, {size_mb:.1f} MB")

    return prefix_embeddings, leaf_embeddings


def embed_poma_chunks(chunksets: list, raw_data: dict, force: bool = False) -> tuple[np.ndarray, dict]:
    """
    Embed individual POMA chunks (for balanced scoring) using pre-computed to_embed field.

    Args:
        chunksets: List of ChunksetItem objects
        raw_data: Dict from load_raw_poma_data: {doc_id: (chunks_list, chunksets_raw)}

    Returns:
        embeddings: np.ndarray of all chunk embeddings
        chunk_map: dict mapping "doc_id:chunkset:N" -> list of embedding indices
    """
    name = "poma_chunks"

    if vectors_exist(name) and not force:
        embeddings, ids, metadata = load_vectors(name)
        chunk_map = json.loads(metadata.get("chunk_map", "{}"))
        print(f"  Loaded {name}: {len(ids)} vectors")
        return embeddings, chunk_map

    # Collect all UNIQUE chunks with their chunkset mapping
    all_texts = []
    all_ids = []
    chunk_map = {}  # "doc_id:chunkset:N" -> list of embedding indices
    chunk_id_to_emb_idx = {}  # "doc_id:chunk:N" -> embedding index (deduplication)

    for cs in chunksets:
        doc_id = cs.doc_id
        if doc_id not in raw_data:
            continue

        chunks_list, chunksets_raw = raw_data[doc_id]

        # Get chunk indices from raw chunkset data
        if cs.chunkset_id < len(chunksets_raw):
            cs_raw = chunksets_raw[cs.chunkset_id]
            cs_chunk_indices = cs_raw.get("chunks", [])

            embedding_indices = []
            for chunk_idx in cs_chunk_indices:
                chunk_id = f"{doc_id}:chunk:{chunk_idx}"

                # Check if this chunk was already embedded (deduplication)
                if chunk_id in chunk_id_to_emb_idx:
                    embedding_indices.append(chunk_id_to_emb_idx[chunk_id])
                elif chunk_idx < len(chunks_list):
                    chunk_to_embed = chunks_list[chunk_idx].get("to_embed", "")
                    if chunk_to_embed and chunk_to_embed.strip():
                        embedding_idx = len(all_texts)
                        all_texts.append(chunk_to_embed)
                        all_ids.append(chunk_id)
                        chunk_id_to_emb_idx[chunk_id] = embedding_idx
                        embedding_indices.append(embedding_idx)

            if embedding_indices:
                chunk_map[f"{doc_id}:chunkset:{cs.chunkset_id}"] = embedding_indices

    print(f"Embedding {len(all_texts)} POMA chunks...")
    embeddings = embed_texts(all_texts)
    save_vectors(name, all_ids, embeddings, extra_metadata={"chunk_map": json.dumps(chunk_map)})
    return embeddings, chunk_map


def embed_databricks_chunks(chunks: list, force: bool = False) -> np.ndarray:
    """Embed Databricks naive chunks. Returns embeddings array."""
    name = "databricks_chunks"

    if vectors_exist(name) and not force:
        embeddings, ids, _ = load_vectors(name)
        print(f"  Loaded {name}: {len(ids)} vectors")
        return embeddings

    print(f"Embedding {len(chunks)} Databricks chunks...")
    texts = [c.text for c in chunks]
    ids = [f"{c.doc_id}:db_chunk:{c.chunk_id}" for c in chunks]
    embeddings = embed_texts(texts)
    save_vectors(name, ids, embeddings)
    return embeddings


def embed_unstructured_elements(elements: list, force: bool = False) -> np.ndarray:
    """Embed Unstructured elements. Returns embeddings array."""
    name = "unstructured_elements"

    if vectors_exist(name) and not force:
        embeddings, ids, _ = load_vectors(name)
        print(f"  Loaded {name}: {len(ids)} vectors")
        return embeddings

    print(f"Embedding {len(elements)} Unstructured elements...")
    texts = [e.text for e in elements]
    ids = [f"{e.doc_id}:us_element:{e.chunk_id}" for e in elements]
    embeddings = embed_texts(texts)
    save_vectors(name, ids, embeddings)
    return embeddings
