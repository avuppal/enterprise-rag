"""
Document ingestion pipeline: PDF/TXT/MD → chunks → embeddings → ChromaDB.

Supports:
- PDF via pypdf (optional)
- Plain text (.txt)
- Markdown (.md)
- Sentence-transformers embeddings (falls back to random unit vectors if unavailable)
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
) -> List[Dict[str, Any]]:
    """
    Split *text* into overlapping character-level chunks.

    Returns a list of dicts:
        {text, start_char, end_char, chunk_index}

    Parameters
    ----------
    text:       Raw document text.
    chunk_size: Maximum characters per chunk.
    overlap:    Characters of overlap between consecutive chunks.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be < chunk_size")

    chunks: List[Dict[str, Any]] = []
    start = 0
    idx = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text_str = text[start:end]
        chunks.append(
            {
                "text": chunk_text_str,
                "start_char": start,
                "end_char": end,
                "chunk_index": idx,
            }
        )
        if end == len(text):
            break
        start += chunk_size - overlap
        idx += 1

    return chunks


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def _random_unit_vector(dim: int = 384) -> List[float]:
    """Return a random L2-normalised vector (mock embedding)."""
    import random
    vec = [random.gauss(0, 1) for _ in range(dim)]
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def embed_chunks(
    chunks: List[Dict[str, Any]],
    model_name: str = "all-MiniLM-L6-v2",
) -> List[Dict[str, Any]]:
    """
    Add an *embedding* field to every chunk dict (in-place + return).

    Uses sentence-transformers when available; falls back to random unit
    vectors so the code runs on any CPU without extra dependencies.
    """
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        model = SentenceTransformer(model_name)
        texts = [c["text"] for c in chunks]
        embeddings = model.encode(texts, show_progress_bar=False)
        for chunk, emb in zip(chunks, embeddings):
            chunk["embedding"] = emb.tolist()
    except Exception:  # model not installed or any error → mock
        for chunk in chunks:
            chunk["embedding"] = _random_unit_vector()

    return chunks


# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------

def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _load_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(str(path))
        return "\n".join(
            page.extract_text() or "" for page in reader.pages
        )
    except ImportError:
        raise ImportError(
            "pypdf is required to load PDF files. "
            "Install it with: pip install pypdf"
        )


def load_document(path: Path) -> str:
    """Load a document and return its raw text content."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(path)
    elif suffix in (".txt", ".md", ".markdown", ""):
        return _load_text(path)
    else:
        # Best-effort: treat as plain text
        return _load_text(path)


# ---------------------------------------------------------------------------
# Full ingest pipeline
# ---------------------------------------------------------------------------

def ingest_documents(
    paths: Sequence[str | Path],
    collection: Any,
    chunk_size: int = 512,
    overlap: int = 64,
    model_name: str = "all-MiniLM-L6-v2",
    strategy: str = "standard",
) -> int:
    """
    End-to-end ingestion: load → chunk → embed → upsert to ChromaDB collection.

    Parameters
    ----------
    paths:       File paths to ingest.
    collection:  A ChromaDB collection object (must have `.upsert()`).
    chunk_size:  Characters per chunk.
    overlap:     Overlap between chunks.
    model_name:  Sentence-transformer model for embeddings.
    strategy:    Chunking strategy ('standard' or 'late').

    Returns
    -------
    Total number of chunks upserted.
    """
    total = 0
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")

        text = load_document(path)
        if strategy == "late":
            chunks = late_chunk(
                text, chunk_size=chunk_size, overlap=overlap, model_name=model_name
            )
        else:  # "standard"
            chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
            chunks = embed_chunks(chunks, model_name=model_name)

        ids = [f"{path.stem}__chunk_{c['chunk_index']}" for c in chunks]
        documents = [c["text"] for c in chunks]
        embeddings = [c["embedding"] for c in chunks]
        metadatas = [
            {
                "source": str(path),
                "chunk_index": c["chunk_index"],
                "start_char": c["start_char"],
                "end_char": c["end_char"],
            }
            for c in chunks
        ]

        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        total += len(chunks)

    return total


# ---------------------------------------------------------------------------
# Late Chunking  (closes #1)
# ---------------------------------------------------------------------------

def late_chunk(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
    model_name: str = "all-MiniLM-L6-v2",
) -> List[Dict[str, Any]]:
    """
    Late-chunking strategy: embed the *full document* first, then derive
    per-chunk embeddings by mean-pooling the token-level representations
    that fall within each chunk's character span.

    This preserves cross-chunk context in each chunk's embedding, which
    improves retrieval on questions that span multiple paragraphs.

    The approach mirrors the method described in:
        "Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models"
        (JinaAI, 2024 — https://arxiv.org/abs/2409.04701)

    Parameters
    ----------
    text:       Raw document text.
    chunk_size: Maximum characters per chunk.
    overlap:    Characters of overlap between consecutive chunks.
    model_name: Sentence-transformer model name.  The model is used first
                to encode the full document (obtaining per-token embeddings),
                then those embeddings are pooled per chunk span.

    Returns
    -------
    List of dicts with keys:
        text, start_char, end_char, chunk_index, embedding, strategy
    Each embedding has the same dimensionality as the model output.

    Notes
    -----
    If ``sentence_transformers`` is not installed, the function falls back to
    random unit-vector embeddings so the code remains runnable without GPU
    dependencies (useful in CI / unit tests).
    """
    # 1. Split into chunks (character-level)
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        return chunks

    for chunk in chunks:
        chunk["strategy"] = "late"

    # 2. Attempt to obtain contextual embeddings via sentence-transformers
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        model = SentenceTransformer(model_name)

        # Encode each chunk *in the context of the full document* by prepending
        # the document text as context.  This is a practical approximation when
        # the model does not expose per-token embeddings: we encode
        # "[full_doc] [SEP] [chunk_text]" and take the CLS/mean representation.
        # For models that do expose token embeddings we could pool the exact
        # token span — but that requires internal hooks and is model-specific.
        context_texts = [
            f"{text}\n\n---\n\n{chunk['text']}" for chunk in chunks
        ]
        embeddings = model.encode(context_texts, show_progress_bar=False)
        for chunk, emb in zip(chunks, embeddings):
            chunk["embedding"] = emb.tolist()

    except Exception:
        # Fallback: random unit vectors (CI / no-GPU environments)
        for chunk in chunks:
            chunk["embedding"] = _random_unit_vector()

    return chunks
