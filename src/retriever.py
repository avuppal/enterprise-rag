"""
Hybrid retrieval: dense (ChromaDB) + sparse (BM25) with Reciprocal Rank Fusion.

Components
----------
BM25Retriever   — Pure-Python BM25 (Okapi BM25); no external library.
DenseRetriever  — Thin wrapper around a ChromaDB collection.
HybridRetriever — Merges dense + sparse via RRF and re-scores with alpha.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _cosine_sim(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a)) or 1e-10
    norm_b = math.sqrt(sum(y * y for y in b)) or 1e-10
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# BM25Retriever
# ---------------------------------------------------------------------------

class BM25Retriever:
    """
    Okapi BM25 retriever backed by an in-memory corpus.

    BM25 score formula (per document d for query q):
        score(d, q) = Σ IDF(t) * TF_norm(t, d)

    where:
        IDF(t)       = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)
        TF_norm(t,d) = tf * (k1+1) / (tf + k1*(1 - b + b*dl/avgdl))

    Parameters
    ----------
    corpus: list of dicts with at least a ``text`` key.
            Extra keys are preserved in results (``source``, ``chunk_index``, …).
    k1:     Term-frequency saturation parameter (default 1.5).
    b:      Length normalisation parameter (default 0.75).
    """

    def __init__(
        self,
        corpus: List[Dict[str, Any]],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.corpus = corpus
        self.k1 = k1
        self.b = b

        # Pre-tokenize corpus
        self._tokenized: List[List[str]] = [
            _tokenize(doc["text"]) for doc in corpus
        ]
        self._dl: List[int] = [len(toks) for toks in self._tokenized]
        self._avgdl: float = (
            sum(self._dl) / len(self._dl) if self._dl else 1.0
        )
        self._N: int = len(corpus)

        # Document frequency per term
        self._df: Dict[str, int] = defaultdict(int)
        for toks in self._tokenized:
            for term in set(toks):
                self._df[term] += 1

    # ------------------------------------------------------------------
    def _idf(self, term: str) -> float:
        df = self._df.get(term, 0)
        return math.log((self._N - df + 0.5) / (df + 0.5) + 1)

    def _score(self, query_terms: List[str], doc_idx: int) -> float:
        toks = self._tokenized[doc_idx]
        dl = self._dl[doc_idx]
        tf_map: Dict[str, int] = defaultdict(int)
        for t in toks:
            tf_map[t] += 1

        score = 0.0
        for term in query_terms:
            tf = tf_map.get(term, 0)
            if tf == 0:
                continue
            idf = self._idf(term)
            tf_norm = (
                tf
                * (self.k1 + 1)
                / (tf + self.k1 * (1 - self.b + self.b * dl / self._avgdl))
            )
            score += idf * tf_norm
        return score

    # ------------------------------------------------------------------
    def query(self, query_text: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Return top-k results sorted by BM25 score (descending).

        Each result dict contains:
            {text, score, source, chunk_index, …}
        """
        q_terms = _tokenize(query_text)
        scored = [
            (self._score(q_terms, i), i) for i in range(self._N)
        ]
        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, idx in scored[:top_k]:
            doc = dict(self.corpus[idx])
            doc["score"] = score
            doc.setdefault("source", "")
            doc.setdefault("chunk_index", idx)
            results.append(doc)
        return results


# ---------------------------------------------------------------------------
# DenseRetriever
# ---------------------------------------------------------------------------

class DenseRetriever:
    """
    Dense retriever backed by a ChromaDB collection.

    Parameters
    ----------
    collection:  ChromaDB collection with `.query()` method.
    embed_fn:    Optional callable ``(text: str) -> List[float]``.
                 If None, passes ``query_texts`` to ChromaDB directly.
    """

    def __init__(
        self,
        collection: Any,
        embed_fn: Optional[Any] = None,
    ) -> None:
        self.collection = collection
        self.embed_fn = embed_fn

    def query(self, query_text: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Query ChromaDB and return top-k results.

        Returns list of dicts: {text, score, source, chunk_index}
        """
        if self.embed_fn is not None:
            embedding = self.embed_fn(query_text)
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
        else:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        output = []
        for doc, meta, dist in zip(docs, metas, dists):
            # ChromaDB returns L2 distances; convert to similarity score
            similarity = 1.0 / (1.0 + dist) if dist is not None else 0.0
            output.append(
                {
                    "text": doc,
                    "score": similarity,
                    "source": meta.get("source", ""),
                    "chunk_index": meta.get("chunk_index", -1),
                }
            )
        return output


# ---------------------------------------------------------------------------
# HybridRetriever (Reciprocal Rank Fusion)
# ---------------------------------------------------------------------------

class HybridRetriever:
    """
    Hybrid retriever that fuses dense and sparse results via RRF.

    Reciprocal Rank Fusion (RRF):
        rrf_score(d) = Σ_r  1 / (k + rank_r(d))

    where k=60 is a smoothing constant and the sum is over each ranked list.
    The fused score is then blended with ``alpha`` to weight dense vs. sparse.

    Parameters
    ----------
    dense:  DenseRetriever instance.
    bm25:   BM25Retriever instance.
    alpha:  Weight for dense scores (0 = pure BM25, 1 = pure dense).
            In RRF mode this controls which list gets higher weight.
    rrf_k:  RRF smoothing constant (default 60).
    """

    def __init__(
        self,
        dense: DenseRetriever,
        bm25: BM25Retriever,
        alpha: float = 0.5,
        rrf_k: int = 60,
    ) -> None:
        self.dense = dense
        self.bm25 = bm25
        self.alpha = alpha
        self.rrf_k = rrf_k

    # ------------------------------------------------------------------
    def _rrf_score(
        self,
        dense_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Merge two ranked lists with RRF and return sorted combined list."""
        scores: Dict[str, float] = defaultdict(float)
        doc_map: Dict[str, Dict[str, Any]] = {}

        for rank, doc in enumerate(dense_results):
            key = f"{doc.get('source', '')}::{doc.get('chunk_index', rank)}"
            scores[key] += self.alpha * (1.0 / (self.rrf_k + rank + 1))
            doc_map[key] = doc

        for rank, doc in enumerate(bm25_results):
            key = f"{doc.get('source', '')}::{doc.get('chunk_index', rank)}"
            scores[key] += (1 - self.alpha) * (
                1.0 / (self.rrf_k + rank + 1)
            )
            if key not in doc_map:
                doc_map[key] = doc

        merged = []
        for key, fused_score in sorted(
            scores.items(), key=lambda x: x[1], reverse=True
        ):
            doc = dict(doc_map[key])
            doc["score"] = fused_score
            merged.append(doc)

        return merged

    def query(self, query_text: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Return top-k hybrid-fused results."""
        dense_res = self.dense.query(query_text, top_k=top_k)
        bm25_res = self.bm25.query(query_text, top_k=top_k)
        merged = self._rrf_score(dense_res, bm25_res)
        return merged[:top_k]
