"""
Hash-based incremental embedding utilities.

Vector files are named: {doc_name}_{hash}.npz
where hash = MD5[:8] of the source file.

This allows:
- Skipping unchanged documents (hash matches)
- Detecting changed documents (hash differs)
- Adding new documents (no vector file exists)
"""
import hashlib
import os
from pathlib import Path
from typing import Optional


def md5_file(path: str | Path, length: int = 8) -> str:
    """Compute first N chars of MD5 hash of a file."""
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()[:length]


def find_vector_file(vectors_dir: Path, doc_name: str, expected_hash: str) -> Optional[Path]:
    """
    Find vector file for a document.

    Returns:
        Path if {doc_name}_{expected_hash}.npz exists
        None if not found (needs embedding)
    """
    target = vectors_dir / f"{doc_name}_{expected_hash}.npz"
    if target.exists():
        return target
    return None


def find_any_vector_file(vectors_dir: Path, doc_name: str) -> Optional[tuple[Path, str]]:
    """
    Find any vector file for a document (any hash).

    Returns:
        (path, hash) if found
        None if not found
    """
    if not vectors_dir.exists():
        return None

    for f in vectors_dir.iterdir():
        if f.name.startswith(doc_name + "_") and f.suffix == ".npz":
            # Extract hash from filename
            hash_part = f.stem.split("_")[-1]
            return f, hash_part
    return None


def cleanup_old_vectors(vectors_dir: Path, doc_name: str, keep_hash: str):
    """Remove old vector files for a document (different hashes)."""
    if not vectors_dir.exists():
        return

    for f in vectors_dir.iterdir():
        if f.name.startswith(doc_name + "_") and f.suffix == ".npz":
            hash_part = f.stem.split("_")[-1]
            if hash_part != keep_hash:
                print(f"    Removing old: {f.name}")
                f.unlink()


def get_docs_needing_embedding(
    source_dir: Path,
    vectors_dir: Path,
    source_ext: str,
    doc_names: list[str]
) -> list[tuple[str, str, Path]]:
    """
    Determine which documents need embedding.

    Returns list of (doc_name, hash, source_path) for docs that need embedding.
    """
    needs_embedding = []

    for doc_name in doc_names:
        source_path = source_dir / f"{doc_name}{source_ext}"
        if not source_path.exists():
            continue

        current_hash = md5_file(source_path)
        existing = find_vector_file(vectors_dir, doc_name, current_hash)

        if existing:
            # Already embedded with matching hash
            continue

        # Needs embedding (new or changed)
        needs_embedding.append((doc_name, current_hash, source_path))

        # Check if there's an old version
        old = find_any_vector_file(vectors_dir, doc_name)
        if old:
            old_path, old_hash = old
            print(f"  {doc_name}: hash changed {old_hash} -> {current_hash}")
        else:
            print(f"  {doc_name}: new document")

    return needs_embedding
