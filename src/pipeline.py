"""
End-to-end RAG pipeline: retrieve → rerank → prompt → generate.

RAGPipeline
-----------
Fully injectable: bring your own retriever, reranker, and LLM function.
No API keys, no hardcoded models.  Works out-of-the-box with any callable LLM.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional


DEFAULT_PROMPT_TEMPLATE = """You are a helpful assistant. Answer the question using ONLY the context below.
If the context does not contain enough information, say "I don't know."

Context:
{context}

Question: {question}

Answer:"""


class RAGPipeline:
    """
    Production-grade RAG pipeline.

    Parameters
    ----------
    retriever:    Any object with a ``.query(text, top_k) -> List[dict]`` method.
                  Can be BM25Retriever, DenseRetriever, or HybridRetriever.
    reranker:     Optional callable ``(query, candidates, top_n) -> List[dict]``.
                  If None, top-k retrieved chunks are used directly.
    llm_fn:       Callable ``(prompt: str) -> str``.
                  Fully injectable — pass any LLM wrapper here.
    prompt_template: Format string with ``{context}`` and ``{question}`` placeholders.

    Example
    -------
    >>> pipeline = RAGPipeline(retriever=hybrid, reranker=cross_encoder_rerank, llm_fn=openai_fn)
    >>> result = pipeline.query("What is RAG?", top_k=10, rerank_top_n=3)
    >>> print(result["answer"])
    """

    def __init__(
        self,
        retriever: Any,
        llm_fn: Callable[[str], str],
        reranker: Optional[Callable[..., List[Dict[str, Any]]]] = None,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker
        self.llm_fn = llm_fn
        self.prompt_template = prompt_template

    # ------------------------------------------------------------------
    def _build_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Concatenate chunk texts into a single context string."""
        parts = []
        for i, chunk in enumerate(chunks, start=1):
            source = chunk.get("source", "unknown")
            text = chunk.get("text", "")
            parts.append(f"[{i}] (source: {source})\n{text}")
        return "\n\n".join(parts)

    def _build_prompt(self, question: str, chunks: List[Dict[str, Any]]) -> str:
        context = self._build_context(chunks)
        return self.prompt_template.format(context=context, question=question)

    def _extract_sources(self, chunks: List[Dict[str, Any]]) -> List[str]:
        seen: set = set()
        sources = []
        for c in chunks:
            src = c.get("source", "")
            if src and src not in seen:
                seen.add(src)
                sources.append(src)
        return sources

    # ------------------------------------------------------------------
    def query(
        self,
        question: str,
        top_k: int = 10,
        rerank_top_n: int = 3,
    ) -> Dict[str, Any]:
        """
        Run the full RAG pipeline for a single question.

        Returns
        -------
        dict with keys:
            answer          — LLM-generated answer string
            sources         — deduplicated list of source paths
            retrieved_chunks — raw retrieval results (list of dicts)
            reranked_chunks  — post-reranking results (list of dicts)
            latency_ms       — per-stage timing dict
        """
        timings: Dict[str, float] = {}

        # Stage 1: Retrieve
        t0 = time.perf_counter()
        retrieved = self.retriever.query(question, top_k=top_k)
        timings["retrieve_ms"] = (time.perf_counter() - t0) * 1000

        # Stage 2: Rerank (optional)
        t1 = time.perf_counter()
        if self.reranker is not None and retrieved:
            reranked = self.reranker(question, retrieved, rerank_top_n)
        else:
            reranked = retrieved[:rerank_top_n]
        timings["rerank_ms"] = (time.perf_counter() - t1) * 1000

        # Stage 3: Build prompt + generate
        t2 = time.perf_counter()
        prompt = self._build_prompt(question, reranked)
        timings["prompt_build_ms"] = (time.perf_counter() - t2) * 1000

        t3 = time.perf_counter()
        answer = self.llm_fn(prompt)
        timings["generate_ms"] = (time.perf_counter() - t3) * 1000

        timings["total_ms"] = sum(timings.values())

        return {
            "answer": answer,
            "sources": self._extract_sources(reranked),
            "retrieved_chunks": retrieved,
            "reranked_chunks": reranked,
            "latency_ms": timings,
        }

    # ------------------------------------------------------------------
    def batch_query(
        self,
        questions: List[str],
        top_k: int = 10,
        rerank_top_n: int = 3,
    ) -> List[Dict[str, Any]]:
        """Run pipeline over a list of questions and return list of result dicts."""
        return [self.query(q, top_k=top_k, rerank_top_n=rerank_top_n) for q in questions]
