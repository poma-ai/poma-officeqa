"""
Data loaders: POMA corpus, naive chunks, unstructured elements, questions.
"""

import os
import re
import csv
import json
import glob
import zipfile
from pathlib import Path

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    RecursiveCharacterTextSplitter = None

from .config import (
    NAIVE_SEPARATORS, NAIVE_CHUNK_SIZE, NAIVE_CHUNK_OVERLAP,
    TRANSFORMED_DIR, UNSTRUCTURED_DIR,
)
from .models import ChunksetItem, NaiveChunk, OfficeQAQuestion
from .context import n_tokens


# =====================================================================
# Utility Functions
# =====================================================================

def strip_extension(filename: str) -> str:
    return os.path.splitext(filename)[0]


# =====================================================================
# POMA Corpus Loading
# =====================================================================

def list_poma_files(poma_dir: str) -> list[str]:
    """List all .poma files in directory."""
    files = sorted(glob.glob(os.path.join(poma_dir, "*.poma")))
    if not files:
        raise RuntimeError(f"No .poma files found in {poma_dir}")
    return files


def get_poma_filename(zf: zipfile.ZipFile, poma_path: str = None) -> str | None:
    """Extract filename from POMA archive metadata."""
    filename_heuristic = None
    if poma_path:
        basename = os.path.basename(poma_path)
        name_without_ext = strip_extension(basename)
        if name_without_ext.endswith('.pdf'):
            name_without_ext = strip_extension(name_without_ext)
        if len(name_without_ext) < 50 and not (len(name_without_ext) == 36 and name_without_ext.count('-') == 4):
            filename_heuristic = name_without_ext
    
    metadata_result = None
    try:
        comment = zf.comment.decode("utf-8", errors="replace")
        meta = json.loads(comment)
        source_file = meta.get("source_file")
        if source_file:
            metadata_result = strip_extension(source_file)
    except Exception:
        pass
    
    if not metadata_result:
        try:
            meta = json.loads(zf.read("metadata_content.json").decode("utf-8", errors="replace"))
            filename = meta.get("filename")
            if filename:
                metadata_result = strip_extension(filename)
        except Exception:
            pass
    
    if filename_heuristic and metadata_result:
        if 'account_' in metadata_result.lower() and 'treasury_bulletin' in filename_heuristic.lower():
            return filename_heuristic
        if (len(metadata_result) > 50 or (len(metadata_result) == 36 and metadata_result.count('-') == 4)) and len(filename_heuristic) < 50:
            return filename_heuristic
    
    return metadata_result if metadata_result else filename_heuristic


def load_chunksets_from_poma(zf: zipfile.ZipFile, doc_id: str) -> list[ChunksetItem]:
    """Load chunksets from a POMA archive."""
    chunks_raw = json.loads(zf.read("chunks.json").decode("utf-8", errors="replace"))
    chunksets = json.loads(zf.read("chunksets.json").decode("utf-8", errors="replace"))
    chunks_map = {int(c["chunk_index"]): str(c.get("content", "") or "") for c in chunks_raw}
    depth_map = {int(c["chunk_index"]): int(c.get("depth", 0)) for c in chunks_raw}

    out: list[ChunksetItem] = []
    for cs in chunksets:
        cs_id = int(cs.get("chunkset_index", 0))
        idxs = [int(i) for i in cs.get("chunks", [])]
        path_texts = [chunks_map.get(i, "") for i in idxs]
        path_depths = [depth_map.get(i, 0) for i in idxs]

        leaf_idx = cs.get("leaf_idx")
        canonical_ancestors = cs.get("canonical_ancestors")
        parent_map_raw = cs.get("parent_map")
        parent_map = None
        if parent_map_raw:
            parent_map = {int(k): (int(v) if v is not None else None) for k, v in parent_map_raw.items()}

        out.append(ChunksetItem(
            doc_id=doc_id,
            chunkset_id=cs_id,
            contents=str(cs.get("contents", "") or ""),
            to_embed=str(cs.get("to_embed", "") or ""),
            path_texts=path_texts,
            path_indices=idxs,
            path_depths=path_depths,
            leaf_idx=leaf_idx,
            canonical_ancestors=canonical_ancestors,
            parent_map=parent_map,
        ))
    return out


def load_poma_corpus(poma_dir: str) -> tuple[list[ChunksetItem], dict[str, str]]:
    """Load chunksets and build filename -> doc_id mapping."""
    chunksets: list[ChunksetItem] = []
    filename_to_docid: dict[str, str] = {}

    for i, path in enumerate(list_poma_files(poma_dir)):
        doc_id = str(i)
        with zipfile.ZipFile(path, "r") as zf:
            filename = get_poma_filename(zf, poma_path=path)
            if filename:
                filename_to_docid[filename] = doc_id
            chunksets.extend(load_chunksets_from_poma(zf, doc_id))
        n_cs = sum(1 for cs in chunksets if cs.doc_id == doc_id)
        print(f"Loaded doc {doc_id}: filename={filename or 'N/A'} chunksets={n_cs}")

    return chunksets, filename_to_docid


def load_raw_poma_data(poma_dir: str, filename_to_docid: dict[str, str]) -> dict[str, tuple[list[dict], list[dict]]]:
    """Load raw chunks and chunksets dicts from each poma file (for generate_cheatsheets)."""
    raw_data: dict[str, tuple[list[dict], list[dict]]] = {}
    for path in list_poma_files(poma_dir):
        with zipfile.ZipFile(path, "r") as zf:
            filename = get_poma_filename(zf, poma_path=path)
            if filename and filename in filename_to_docid:
                doc_id = filename_to_docid[filename]
                chunks = json.loads(zf.read("chunks.json").decode("utf-8", errors="replace"))
                chunksets = json.loads(zf.read("chunksets.json").decode("utf-8", errors="replace"))
                raw_data[doc_id] = (chunks, chunksets)
    return raw_data


# =====================================================================
# Naive Chunk Loading
# =====================================================================

def load_naive_chunks_from_txt(poma_dir: str, filename_to_docid: dict[str, str], 
                                transformed_dir: Path = TRANSFORMED_DIR) -> list[NaiveChunk]:
    """Load source texts from transformed directory and chunk with RecursiveCharacterTextSplitter."""
    if RecursiveCharacterTextSplitter is None:
        print("WARNING: langchain_text_splitters not installed, naive chunking unavailable")
        return []

    splitter = RecursiveCharacterTextSplitter(
        separators=NAIVE_SEPARATORS,
        chunk_size=NAIVE_CHUNK_SIZE,
        chunk_overlap=NAIVE_CHUNK_OVERLAP,
        length_function=n_tokens,
    )

    naive_chunks: list[NaiveChunk] = []
    chunk_id = 0

    for source_name, doc_id in filename_to_docid.items():
        txt_path = Path(transformed_dir) / f"{source_name}.txt"

        if not txt_path.exists():
            print(f"  Warning: transformed file not found: {txt_path}")
            continue

        try:
            with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
                source_text = f.read()
        except Exception as e:
            print(f"  Warning: could not read {txt_path}: {e}")
            continue

        chunks = splitter.split_text(source_text)
        for text in chunks:
            naive_chunks.append(NaiveChunk(doc_id=doc_id, chunk_id=chunk_id, text=text))
            chunk_id += 1

        print(f"  Loaded doc {doc_id} ({source_name}): {len(chunks)} chunks from .txt")

    print(f"Loaded {len(naive_chunks)} databricks_rcs chunks from {transformed_dir}")
    return naive_chunks


# =====================================================================
# Unstructured Elements Loading
# =====================================================================

def load_unstructured_elements(filename_to_docid: dict[str, str], 
                                unstructured_dir: Path = UNSTRUCTURED_DIR) -> list[NaiveChunk]:
    """Load raw Unstructured.io elements as chunks (NO rechunking)."""
    chunks: list[NaiveChunk] = []
    chunk_id = 0
    
    for source_name, doc_id in filename_to_docid.items():
        json_path = Path(unstructured_dir) / f"{source_name}.pdf.json"
        
        if not json_path.exists():
            json_path = Path(unstructured_dir) / f"{source_name}.json"
            if not json_path.exists():
                continue
        
        try:
            with open(json_path, "r", encoding="utf-8", errors="replace") as f:
                data = json.load(f)
            
            elements = data if isinstance(data, list) else data.get("elements", [])
            doc_chunks = 0
            for el in elements:
                text = el.get("text", "")
                if text.strip():
                    chunks.append(NaiveChunk(doc_id=doc_id, chunk_id=chunk_id, text=text))
                    chunk_id += 1
                    doc_chunks += 1
            
            print(f"  Loaded doc {doc_id} ({source_name}): {doc_chunks} elements from unstructured")
                
        except Exception as e:
            print(f"  Warning: could not read {json_path}: {e}")
            continue
    
    print(f"Loaded {len(chunks)} unstructured elements from {unstructured_dir}")
    return chunks


# =====================================================================
# OfficeQA Questions Loading
# =====================================================================

def load_officeqa(csv_path: str, filename_to_docid: dict[str, str]) -> list[OfficeQAQuestion]:
    """Load questions from OfficeQA CSV."""
    questions: list[OfficeQAQuestion] = []
    skipped = 0
    skipped_partial = 0

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            source_files_raw = row.get("source_files", "")
            source_files = [strip_extension(s.strip()) for s in source_files_raw.split("\n") if s.strip()]

            if not source_files:
                skipped += 1
                continue

            all_present = all(sf in filename_to_docid for sf in source_files)
            any_present = any(sf in filename_to_docid for sf in source_files)

            if all_present:
                matched_file = source_files[0]
                matched_doc_id = filename_to_docid[matched_file]
                questions.append(OfficeQAQuestion(
                    uid=row.get("uid", ""),
                    question=row.get("question", ""),
                    answer=row.get("answer", ""),
                    source_file=matched_file,
                    difficulty=row.get("difficulty", ""),
                    doc_id=matched_doc_id,
                ))
            elif any_present:
                skipped_partial += 1
            else:
                skipped += 1

    print(f"Loaded {len(questions)} questions (skipped {skipped} no match, {skipped_partial} partial match)")
    return questions
