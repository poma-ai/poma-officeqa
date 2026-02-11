#!/usr/bin/env python3
"""
Embed all corpus items once and for all.

Usage:
    python embed_all.py           # Embed only missing
    python embed_all.py --force   # Re-embed everything
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from bench.loaders import (
    load_poma_corpus, load_raw_poma_data, load_officeqa,
    load_naive_chunks_from_txt, load_unstructured_elements
)
from bench.vectors import (
    embed_questions, embed_poma_chunksets, embed_poma_chunks,
    embed_databricks_chunks, embed_unstructured_elements,
    list_vector_files
)


def main():
    parser = argparse.ArgumentParser(description="Embed all corpus items")
    parser.add_argument("--force", action="store_true", help="Re-embed everything")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    poma_dir = script_dir / "data" / "poma"

    print("=" * 60)
    print("EMBEDDING ALL CORPUS ITEMS")
    print("=" * 60)

    # Load corpus
    print("\n[1/6] Loading corpus...")
    chunksets, filename_to_docid = load_poma_corpus(poma_dir)
    raw_data = load_raw_poma_data(poma_dir, filename_to_docid)
    questions = load_officeqa(script_dir / "data" / "officeqa.csv", filename_to_docid)
    db_chunks = load_naive_chunks_from_txt(poma_dir, filename_to_docid)
    us_elements = load_unstructured_elements(filename_to_docid)

    print(f"  {len(chunksets):,} POMA chunksets")
    print(f"  {len(questions)} questions")
    print(f"  {len(db_chunks):,} Databricks chunks")
    print(f"  {len(us_elements):,} Unstructured elements")

    # Embed everything
    print("\n[2/6] Questions...")
    embed_questions(questions, force=args.force)

    print("\n[3/6] POMA chunksets (full contents)...")
    embed_poma_chunksets(chunksets, force=args.force)

    print("\n[4/6] POMA chunks (for balanced scoring)...")
    embed_poma_chunks(chunksets, raw_data, force=args.force)

    print("\n[5/6] Databricks chunks...")
    embed_databricks_chunks(db_chunks, force=args.force)

    print("\n[6/6] Unstructured elements...")
    embed_unstructured_elements(us_elements, force=args.force)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    files = list_vector_files()
    total_vectors = 0
    total_size = 0
    for f in files:
        if "error" not in f:
            print(f"  {f['file']}: {f['count']:,} vectors, {f['size_mb']:.1f} MB")
            total_vectors += f["count"]
            total_size += f["size_mb"]
    print(f"\n  TOTAL: {total_vectors:,} vectors, {total_size:.1f} MB")
    print("=" * 60)


if __name__ == "__main__":
    main()
