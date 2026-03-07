"""
Tests for src/pipeline.py — end-to-end RAG pipeline with mocked retriever + LLM.
"""

import sys
import types
import math
import pytest


# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------

class MockRetriever:
    """Returns canned results for any query."""

    DOCS = [
        {"text": "RAG stands for Retrieval-Augmented Generation.", "source": "intro.txt", "chunk_index": 0, "score": 0.9},
        {"text": "Dense retrieval uses embedding similarity.", "source": "dense.txt", "chunk_index": 0, "score": 0.8},
        {"text": "BM25 is a sparse retrieval method.", "source": "sparse.txt", "chunk_index": 0, "score": 0.7},
        {"text": "Re-ranking improves precision.", "source": "rerank.txt", "chunk_index": 0, "score": 0.6},
        {"text": "Evaluation requires recall and MRR metrics.", "source": "eval.txt", "chunk_index": 0, "score": 0.5},
    ]

    def query(self, text, top_k=10):
        return self.DOCS[:top_k]


def mock_llm(prompt: str) -> str:
    return f"Answer based on context length {len(prompt)}."


def mock_reranker(query, candidates, top_n):
    # Return first top_n, add rerank_score
    result = []
    for i, c in enumerate(candidates[:top_n]):
        d = dict(c)
        d["rerank_score"] = 1.0 / (i + 1)
        result.append(d)
    return result


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------

from src.pipeline import RAGPipeline


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRAGPipeline:
    def setup_method(self):
        self.pipeline = RAGPipeline(
            retriever=MockRetriever(),
            llm_fn=mock_llm,
            reranker=mock_reranker,
        )

    def test_query_returns_dict(self):
        result = self.pipeline.query("What is RAG?")
        assert isinstance(result, dict)

    def test_answer_key_present(self):
        result = self.pipeline.query("What is RAG?")
        assert "answer" in result
        assert isinstance(result["answer"], str)

    def test_sources_key_present(self):
        result = self.pipeline.query("Explain RAG")
        assert "sources" in result
        assert isinstance(result["sources"], list)

    def test_retrieved_chunks_present(self):
        result = self.pipeline.query("What is BM25?")
        assert "retrieved_chunks" in result
        assert isinstance(result["retrieved_chunks"], list)
        assert len(result["retrieved_chunks"]) > 0

    def test_reranked_chunks_present(self):
        result = self.pipeline.query("How does re-ranking work?")
        assert "reranked_chunks" in result
        assert isinstance(result["reranked_chunks"], list)

    def test_rerank_top_n_respected(self):
        result = self.pipeline.query("test", top_k=5, rerank_top_n=2)
        assert len(result["reranked_chunks"]) <= 2

    def test_latency_ms_reported(self):
        result = self.pipeline.query("latency test")
        assert "latency_ms" in result
        lat = result["latency_ms"]
        assert "retrieve_ms" in lat
        assert "rerank_ms" in lat
        assert "generate_ms" in lat
        assert "total_ms" in lat

    def test_latency_positive(self):
        result = self.pipeline.query("timing check")
        for key, val in result["latency_ms"].items():
            assert val >= 0, f"{key} should be >= 0"

    def test_sources_deduplicated(self):
        """Sources list should contain unique paths."""
        result = self.pipeline.query("dedup test", top_k=5, rerank_top_n=5)
        sources = result["sources"]
        assert len(sources) == len(set(sources))

    def test_llm_fn_receives_prompt_with_context(self):
        """Prompt sent to LLM should include chunk text."""
        captured = []

        def capture_llm(prompt):
            captured.append(prompt)
            return "captured"

        p = RAGPipeline(
            retriever=MockRetriever(),
            llm_fn=capture_llm,
            reranker=mock_reranker,
        )
        p.query("RAG question", rerank_top_n=2)
        assert len(captured) == 1
        assert "context" in captured[0].lower() or len(captured[0]) > 50

    def test_no_reranker_uses_top_k(self):
        """Without a reranker, reranked_chunks should be top retrieved chunks."""
        p = RAGPipeline(
            retriever=MockRetriever(),
            llm_fn=mock_llm,
            reranker=None,
        )
        result = p.query("test", top_k=5, rerank_top_n=3)
        assert len(result["reranked_chunks"]) == 3

    def test_custom_prompt_template(self):
        custom_template = "Q: {question}\nCTX: {context}\nA:"
        p = RAGPipeline(
            retriever=MockRetriever(),
            llm_fn=lambda prompt: prompt,  # echo prompt as answer
            reranker=None,
            prompt_template=custom_template,
        )
        result = p.query("What is retrieval?", rerank_top_n=1)
        assert "Q: What is retrieval?" in result["answer"]

    def test_empty_retriever(self):
        """Pipeline should handle empty retrieval gracefully."""

        class EmptyRetriever:
            def query(self, text, top_k=10):
                return []

        p = RAGPipeline(
            retriever=EmptyRetriever(),
            llm_fn=mock_llm,
            reranker=None,
        )
        result = p.query("anything")
        assert result["answer"] is not None
        assert result["retrieved_chunks"] == []

    def test_batch_query(self):
        results = self.pipeline.batch_query(["q1", "q2", "q3"])
        assert len(results) == 3
        for r in results:
            assert "answer" in r

    def test_answer_is_string(self):
        result = self.pipeline.query("string test")
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 0
