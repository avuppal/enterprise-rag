"""
Re-ranking module: cross-encoder scoring + Maximum Marginal Relevance (MMR).

cross_encoder_rerank  — CPU-safe mock using cosine similarity of embeddings.
                        Drop-in for a real cross-encoder (e.g. ms-marco-MiniLM).
mmr_select            — Maximal Marginal Relevance for diversity-aware selection.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(v: List[float]) -> float:
    return math.sqrt(sum(x * x for x in v)) or 1e-10


def cosine_sim(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two equal-length vectors."""
    return _dot(a, b) / (_norm(a) * _norm(b))


def _get_embedding(doc: Dict[str, Any]) -> Optional[List[float]]:
    """Extract the embedding field from a doc dict if present."""
    return doc.get("embedding") or doc.get("emb")


def _random_unit_vector(dim: int = 384) -> List[float]:
    """Deterministic-ish fallback embedding for a document with no embedding."""
    import random

    rng = random.Random(hash(str(id)) & 0xFFFFFFFF)
    vec = [rng.gauss(0, 1) for _ in range(dim)]
    n = _norm(vec) or 1.0
    return [v / n for v in vec]


# ---------------------------------------------------------------------------
# Embed query text (lightweight)
# ---------------------------------------------------------------------------

def _embed_query(query: str, dim: int = 384) -> List[float]:
    """
    Generate a query embedding.

    Tries sentence-transformers → falls back to a character-hash mock so tests
    never need a network connection.
    """
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        _model_cache: Dict[int, Any] = {}
        model = SentenceTransformer("all-MiniLM-L6-v2")
        emb = model.encode([query], show_progress_bar=False)[0]
        return emb.tolist()
    except Exception:
        # Mock: hash-seeded random unit vector
        import random

        rng = random.Random(hash(query) & 0xFFFFFFFF)
        vec = [rng.gauss(0, 1) for _ in range(dim)]
        n = _norm(vec) or 1.0
        return [v / n for v in vec]


# ---------------------------------------------------------------------------
# Cross-encoder re-ranking (CPU mock)
# ---------------------------------------------------------------------------

def cross_encoder_rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    top_n: int = 3,
    embed_fn: Optional[Callable[[str], List[float]]] = None,
) -> List[Dict[str, Any]]:
    """
    Re-rank *candidates* using cosine similarity between query and chunk
    embeddings.  This is a CPU-safe stand-in for a real cross-encoder.

    A real cross-encoder (e.g. ``cross-encoder/ms-marco-MiniLM-L-6-v2``) is a
    drop-in: just replace the score computation with model.predict([(query, text)]).

    Parameters
    ----------
    query:      Query string.
    candidates: List of chunk dicts (may include an ``embedding`` field).
    top_n:      How many to return.
    embed_fn:   Optional custom embedding function ``(text) -> List[float]``.
                Defaults to the internal lightweight embedder.

    Returns
    -------
    Re-ranked list (highest score first), length ≤ top_n.
    """
    if not candidates:
        return []

    _embed = embed_fn if embed_fn is not None else _embed_query
    query_emb = _embed(query)

    scored: List[tuple[float, Dict[str, Any]]] = []
    for doc in candidates:
        doc_emb = _get_embedding(doc)
        if doc_emb is None:
            # Embed on-the-fly from text
            doc_emb = _embed(doc.get("text", ""))
        score = cosine_sim(query_emb, doc_emb)
        new_doc = dict(doc)
        new_doc["rerank_score"] = score
        scored.append((score, new_doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_n]]


# ---------------------------------------------------------------------------
# Maximum Marginal Relevance (MMR)
# ---------------------------------------------------------------------------

def mmr_select(
    candidates: List[Dict[str, Any]],
    query_embedding: List[float],
    top_n: int = 3,
    lambda_: float = 0.5,
    embed_fn: Optional[Callable[[str], List[float]]] = None,
) -> List[Dict[str, Any]]:
    """
    Select *top_n* candidates using Maximum Marginal Relevance.

    MMR balances relevance to the query with diversity among selected docs:

        MMR(d) = λ · sim(d, q) − (1−λ) · max_{s∈S} sim(d, s)

    where S is the set of already-selected documents.

    Parameters
    ----------
    candidates:      Candidate chunk dicts.
    query_embedding: Pre-computed query embedding vector.
    top_n:           Number of results to select.
    lambda_:         Trade-off between relevance (1.0) and diversity (0.0).
    embed_fn:        Optional embedding function for docs lacking embeddings.

    Returns
    -------
    Selected docs in order of MMR selection (list of dicts).
    """
    if not candidates:
        return []

    top_n = min(top_n, len(candidates))
    _embed = embed_fn if embed_fn is not None else _embed_query

    # Ensure every candidate has an embedding
    enriched: List[Dict[str, Any]] = []
    for doc in candidates:
        d = dict(doc)
        if _get_embedding(d) is None:
            d["embedding"] = _embed(d.get("text", ""))
        enriched.append(d)

    selected: List[Dict[str, Any]] = []
    remaining = list(enriched)

    while len(selected) < top_n and remaining:
        if not selected:
            # First pick: highest query relevance
            best = max(
                remaining,
                key=lambda d: cosine_sim(query_embedding, _get_embedding(d)),  # type: ignore[arg-type]
            )
        else:
            # Subsequent picks: MMR criterion
            selected_embs = [_get_embedding(s) for s in selected]

            def mmr_score(d: Dict[str, Any]) -> float:
                emb = _get_embedding(d)
                rel = cosine_sim(query_embedding, emb)  # type: ignore[arg-type]
                redundancy = max(
                    cosine_sim(emb, se)  # type: ignore[arg-type]
                    for se in selected_embs
                )
                return lambda_ * rel - (1 - lambda_) * redundancy

            best = max(remaining, key=mmr_score)

        selected.append(best)
        remaining.remove(best)

    return selected
