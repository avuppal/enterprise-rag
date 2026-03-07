"""
Tests for src/eval.py — Recall@K, MRR, NDCG, ROUGE-L, faithfulness.
"""

import math
import pytest

from src.eval import (
    recall_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    rouge_l_score,
    faithfulness_score,
    run_eval,
)


# ---------------------------------------------------------------------------
# recall_at_k
# ---------------------------------------------------------------------------

class TestRecallAtK:
    def test_perfect_recall(self):
        rel = ["a", "b", "c"]
        ret = ["a", "b", "c", "d", "e"]
        assert recall_at_k(rel, ret, k=3) == 1.0

    def test_zero_recall(self):
        rel = ["x", "y"]
        ret = ["a", "b", "c"]
        assert recall_at_k(rel, ret, k=3) == 0.0

    def test_partial_recall(self):
        rel = ["a", "b", "c"]
        ret = ["a", "d", "e"]
        # 1 of 3 relevant found in top-3
        assert abs(recall_at_k(rel, ret, k=3) - 1 / 3) < 1e-9

    def test_k_cutoff_limits_retrieval(self):
        rel = ["a"]
        ret = ["b", "c", "a"]  # "a" is at rank 3
        assert recall_at_k(rel, ret, k=2) == 0.0
        assert recall_at_k(rel, ret, k=3) == 1.0

    def test_empty_relevant(self):
        assert recall_at_k([], ["a", "b"], k=5) == 0.0

    def test_empty_retrieved(self):
        assert recall_at_k(["a", "b"], [], k=5) == 0.0

    def test_k_larger_than_retrieved(self):
        rel = ["a", "b"]
        ret = ["a"]
        assert recall_at_k(rel, ret, k=100) == 0.5

    def test_duplicates_in_retrieved(self):
        """Duplicates in retrieved should not inflate recall."""
        rel = ["a"]
        ret = ["a", "a", "a"]
        assert recall_at_k(rel, ret, k=3) == 1.0


# ---------------------------------------------------------------------------
# mean_reciprocal_rank
# ---------------------------------------------------------------------------

class TestMRR:
    def test_first_rank(self):
        rel = ["a"]
        ret = ["a", "b", "c"]
        assert mean_reciprocal_rank(rel, ret) == 1.0

    def test_second_rank(self):
        rel = ["b"]
        ret = ["a", "b", "c"]
        assert abs(mean_reciprocal_rank(rel, ret) - 0.5) < 1e-9

    def test_third_rank(self):
        rel = ["c"]
        ret = ["a", "b", "c"]
        assert abs(mean_reciprocal_rank(rel, ret) - 1 / 3) < 1e-9

    def test_no_relevant_found(self):
        assert mean_reciprocal_rank(["x"], ["a", "b", "c"]) == 0.0

    def test_empty_relevant(self):
        assert mean_reciprocal_rank([], ["a", "b"]) == 0.0

    def test_empty_retrieved(self):
        assert mean_reciprocal_rank(["a"], []) == 0.0

    def test_multiple_relevant_first_wins(self):
        """MRR uses rank of *first* relevant document."""
        rel = ["b", "a"]
        ret = ["a", "b", "c"]
        # "a" appears first at rank 1 → MRR = 1.0
        assert mean_reciprocal_rank(rel, ret) == 1.0


# ---------------------------------------------------------------------------
# ndcg_at_k
# ---------------------------------------------------------------------------

class TestNDCGAtK:
    def test_perfect_ranking(self):
        rel = ["a", "b", "c"]
        ret = ["a", "b", "c", "d"]
        score = ndcg_at_k(rel, ret, k=3)
        assert abs(score - 1.0) < 1e-9

    def test_zero_ranking(self):
        rel = ["x", "y"]
        ret = ["a", "b", "c"]
        assert ndcg_at_k(rel, ret, k=3) == 0.0

    def test_partial_ranking(self):
        rel = ["a", "b"]
        ret = ["a", "x", "b"]  # a@1, b@3
        score = ndcg_at_k(rel, ret, k=3)
        # DCG = 1/log2(2) + 1/log2(4) = 1 + 0.5
        # IDCG = 1/log2(2) + 1/log2(3) = 1 + 0.631
        dcg = 1.0 / math.log2(2) + 1.0 / math.log2(4)
        idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
        assert abs(score - dcg / idcg) < 1e-9

    def test_empty_relevant(self):
        assert ndcg_at_k([], ["a", "b"], k=5) == 0.0

    def test_k_cutoff(self):
        rel = ["c"]
        ret = ["a", "b", "c", "d"]
        assert ndcg_at_k(rel, ret, k=2) == 0.0  # "c" not in top 2
        assert ndcg_at_k(rel, ret, k=3) > 0.0   # "c" in top 3

    def test_single_relevant_at_rank_1(self):
        assert ndcg_at_k(["a"], ["a"], k=5) == 1.0


# ---------------------------------------------------------------------------
# rouge_l_score
# ---------------------------------------------------------------------------

class TestRougeL:
    def test_identical(self):
        assert abs(rouge_l_score("hello world", "hello world") - 1.0) < 1e-9

    def test_empty_hypothesis(self):
        assert rouge_l_score("hello world", "") == 0.0

    def test_empty_reference(self):
        assert rouge_l_score("", "hello world") == 0.0

    def test_no_overlap(self):
        assert rouge_l_score("hello world", "goodbye moon") == 0.0

    def test_partial_overlap(self):
        score = rouge_l_score("the cat sat on the mat", "the cat is on the mat")
        assert 0 < score < 1.0

    def test_symmetry_not_required(self):
        """ROUGE-L is not symmetric (precision vs recall differ)."""
        s1 = rouge_l_score("hello world", "hello world and more")
        s2 = rouge_l_score("hello world and more", "hello world")
        # Both should be > 0; they may differ
        assert s1 > 0
        assert s2 > 0

    def test_case_insensitive(self):
        assert rouge_l_score("Hello World", "hello world") == 1.0


# ---------------------------------------------------------------------------
# faithfulness_score
# ---------------------------------------------------------------------------

class TestFaithfulness:
    def test_full_overlap(self):
        answer = "RAG stands for retrieval augmented generation"
        chunks = [{"text": "RAG stands for retrieval augmented generation in NLP"}]
        score = faithfulness_score(answer, chunks)
        assert score > 0.8

    def test_no_overlap(self):
        answer = "xyzzy quux quibble"
        chunks = [{"text": "The quick brown fox"}]
        score = faithfulness_score(answer, chunks)
        assert score == 0.0

    def test_empty_answer(self):
        chunks = [{"text": "some context"}]
        assert faithfulness_score("", chunks) == 0.0

    def test_empty_chunks(self):
        assert faithfulness_score("some answer", []) == 0.0

    def test_partial_overlap(self):
        answer = "retrieval and generation"
        chunks = [{"text": "retrieval augmented systems"}]
        score = faithfulness_score(answer, chunks)
        assert 0 < score < 1.0

    def test_multiple_chunks(self):
        answer = "dogs and cats are pets"
        chunks = [
            {"text": "dogs are loyal animals"},
            {"text": "cats are independent pets"},
        ]
        score = faithfulness_score(answer, chunks)
        assert score > 0


# ---------------------------------------------------------------------------
# run_eval
# ---------------------------------------------------------------------------

class TestRunEval:
    def _make_pipeline(self):
        """Build a minimal pipeline that returns predictable results."""
        from src.pipeline import RAGPipeline

        class FixedRetriever:
            DOCS = [
                {"text": "RAG is retrieval augmented generation", "source": "doc1.txt", "chunk_index": 0, "score": 0.9},
                {"text": "Dense retrieval uses vectors", "source": "doc2.txt", "chunk_index": 0, "score": 0.8},
            ]

            def query(self, text, top_k=10):
                return self.DOCS[:top_k]

        return RAGPipeline(
            retriever=FixedRetriever(),
            llm_fn=lambda p: "RAG stands for retrieval augmented generation.",
            reranker=None,
        )

    def test_returns_dict(self):
        pipeline = self._make_pipeline()
        dataset = [
            {
                "question": "What is RAG?",
                "relevant_ids": ["doc1.txt::0"],
                "reference_answer": "RAG is retrieval augmented generation",
            }
        ]
        metrics = run_eval(pipeline, dataset)
        assert isinstance(metrics, dict)

    def test_metric_keys_present(self):
        pipeline = self._make_pipeline()
        dataset = [{"question": "Q1", "relevant_ids": ["doc1.txt::0"]}]
        metrics = run_eval(pipeline, dataset)
        assert "recall_at_k" in metrics
        assert "mrr" in metrics
        assert "ndcg_at_k" in metrics
        assert "faithfulness" in metrics

    def test_num_queries_correct(self):
        pipeline = self._make_pipeline()
        dataset = [
            {"question": "Q1", "relevant_ids": ["doc1.txt::0"]},
            {"question": "Q2", "relevant_ids": ["doc2.txt::0"]},
        ]
        metrics = run_eval(pipeline, dataset)
        assert metrics["num_queries"] == 2.0

    def test_empty_dataset(self):
        pipeline = self._make_pipeline()
        metrics = run_eval(pipeline, [])
        assert metrics == {}

    def test_rouge_l_computed_when_reference_present(self):
        pipeline = self._make_pipeline()
        dataset = [
            {
                "question": "Q1",
                "relevant_ids": ["doc1.txt::0"],
                "reference_answer": "RAG is retrieval augmented generation",
            }
        ]
        metrics = run_eval(pipeline, dataset)
        assert "rouge_l" in metrics
        assert 0 <= metrics["rouge_l"] <= 1.0

    def test_metrics_in_range(self):
        pipeline = self._make_pipeline()
        dataset = [
            {"question": "What is RAG?", "relevant_ids": ["doc1.txt::0"]},
            {"question": "Dense retrieval?", "relevant_ids": ["doc2.txt::0"]},
        ]
        metrics = run_eval(pipeline, dataset)
        for key in ["recall_at_k", "mrr", "ndcg_at_k", "faithfulness"]:
            assert 0.0 <= metrics[key] <= 1.0, f"{key} out of range"
