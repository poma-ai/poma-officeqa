#!/usr/bin/env python3
"""
Split large vector .npz files by document to stay under GitHub's 100MB limit.

ALWAYS run this after embed_all.py --force!

Usage:
    python bench/split_vectors.py
"""
import json
import numpy as np
import os
import sys

VECTORS_DIR = "vectors"

def get_doc_names():
    """Dynamically load document names from data/poma directory."""
    poma_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'poma')
    if os.path.exists(poma_dir):
        names = sorted([f.replace('.poma', '') for f in os.listdir(poma_dir) if f.endswith('.poma')])
        return names
    return []

DOC_NAMES = get_doc_names()

FILES_TO_SPLIT = [
    ("poma_chunksets.npz", "poma_chunksets"),
    ("poma_chunks.npz", "poma_chunks"),
    ("unstructured_elements.npz", "unstructured_elements"),
]


def split_by_doc(source_file: str, target_dir: str) -> int:
    """Split npz file by document based on id prefix (format: doc_idx:type:local_id)

    Preserves metadata including chunk_map for poma_chunks.
    """
    source_path = os.path.join(VECTORS_DIR, source_file)
    target_path = os.path.join(VECTORS_DIR, target_dir)

    if not os.path.exists(source_path):
        print(f"  SKIP: {source_file} not found")
        return 0

    size_mb = os.path.getsize(source_path) / 1024 / 1024
    if size_mb < 100:
        print(f"  SKIP: {source_file} is only {size_mb:.1f}MB (under 100MB limit)")
        return 0

    print(f"\nSplitting {source_file} ({size_mb:.1f}MB) -> {target_dir}/")
    data = np.load(source_path, allow_pickle=True)
    embeddings = data['embeddings']
    ids = data['ids']

    # Load metadata if present
    metadata = {}
    if 'metadata' in data.files:
        metadata = json.loads(str(data['metadata']))

    # Extract chunk_map if present (for poma_chunks)
    chunk_map = {}
    if 'chunk_map' in metadata:
        chunk_map = json.loads(metadata['chunk_map'])
        print(f"  Found chunk_map with {len(chunk_map)} entries")

    # Extract doc_idx from ids (format: "doc_idx:type:local_id")
    doc_indices = np.array([int(str(id_).split(':')[0]) for id_ in ids])

    os.makedirs(target_path, exist_ok=True)

    # Build a mapping from global index to doc_idx for chunk_map re-indexing
    global_to_doc_local = {}
    doc_offsets = {}
    for doc_idx in sorted(set(doc_indices)):
        mask = doc_indices == doc_idx
        doc_offsets[doc_idx] = np.where(mask)[0][0]  # First index for this doc
        global_indices = np.where(mask)[0]
        for local_idx, global_idx in enumerate(global_indices):
            global_to_doc_local[global_idx] = (doc_idx, local_idx)

    total_saved = 0
    for doc_idx in sorted(set(doc_indices)):
        mask = doc_indices == doc_idx
        doc_embeddings = embeddings[mask]
        doc_ids = ids[mask]
        doc_name = DOC_NAMES[doc_idx] if doc_idx < len(DOC_NAMES) else f"doc_{doc_idx}"

        # Build per-document chunk_map with re-indexed local indices
        doc_chunk_map = {}
        if chunk_map:
            # Filter chunk_map entries for this document
            # Keys are like "0:chunkset:123" where 0 is the doc_idx
            for key, global_indices in chunk_map.items():
                key_doc_idx = int(key.split(':')[0])
                if key_doc_idx == doc_idx:
                    # Re-index to local indices for this document
                    local_indices = []
                    for gi in global_indices:
                        if gi in global_to_doc_local:
                            di, li = global_to_doc_local[gi]
                            if di == doc_idx:
                                local_indices.append(li)
                    if local_indices:
                        doc_chunk_map[key] = [int(i) for i in local_indices]  # Convert int64 to int

        # Build per-document metadata
        doc_metadata = {
            'model': metadata.get('model', 'text-embedding-3-large'),
            'dims': int(metadata.get('dims', 1536)),
            'count': int(len(doc_ids)),
            'doc_id': str(doc_idx),
        }
        if doc_chunk_map:
            doc_metadata['chunk_map'] = json.dumps(doc_chunk_map)

        out_path = os.path.join(target_path, f"{doc_name}.npz")
        np.savez_compressed(
            out_path,
            embeddings=doc_embeddings,
            ids=doc_ids,
            metadata=json.dumps(doc_metadata)
        )
        out_size_mb = os.path.getsize(out_path) / 1024 / 1024
        chunk_map_info = f", {len(doc_chunk_map)} chunk_map entries" if doc_chunk_map else ""
        print(f"  {doc_name}.npz: {len(doc_embeddings)} vectors, {out_size_mb:.1f} MB{chunk_map_info}")

        if out_size_mb >= 100:
            print(f"  WARNING: {doc_name}.npz is {out_size_mb:.1f}MB - still over limit!")

        total_saved += len(doc_embeddings)

    # Remove the large source file
    os.remove(source_path)
    print(f"  DELETED: {source_file}")

    return total_saved


def main():
    print("=" * 60)
    print("SPLITTING LARGE VECTOR FILES (>100MB)")
    print("=" * 60)

    total = 0
    for source_file, target_dir in FILES_TO_SPLIT:
        total += split_by_doc(source_file, target_dir)

    if total == 0:
        print("\nNo files needed splitting.")
    else:
        print(f"\n✓ Split {total} vectors total")

    # Verify no files over 100MB
    print("\nVerifying all files under 100MB...")
    over_limit = []
    for root, dirs, files in os.walk(VECTORS_DIR):
        for f in files:
            if f.endswith('.npz'):
                path = os.path.join(root, f)
                size_mb = os.path.getsize(path) / 1024 / 1024
                if size_mb >= 100:
                    over_limit.append((path, size_mb))

    if over_limit:
        print("\n❌ ERROR: Files still over 100MB limit:")
        for path, size in over_limit:
            print(f"  {path}: {size:.1f}MB")
        sys.exit(1)
    else:
        print("✓ All files under 100MB")


if __name__ == "__main__":
    main()
