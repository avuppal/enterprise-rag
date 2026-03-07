"""
Tests for src/reranker.py — cross-encoder re-ranking, MMR diversity.
All tests use CPU mocks; no sentence-transformers download required.
"""

import sys
import types
import math
import pytest


# ---------------------------------------------------------------------------
# Mock sentence_transformers
# ---------------------------------------------------------------------------

if "sentence_transformers" not in sys.modules:
    st_mock = types.ModuleType("sentence_transformers")

    class FakeST:
        def __init__(self, model_name):
            self.dim = 32

        def encode(self, texts, show_progress_bar=False):
            import random
            results = []
            for t in texts:
                rng = random.Random(hash(t) & 0xFFFFFFFF)
                vec = [rng.gauss(0, 1) for _ in range(self.dim)]
                n = math.sqrt(sum(v * v for v in vec)) or 1.0
                results.append([v / n for v in vec])
            return results

    st_mock.SentenceTransformer = FakeST
    sys.modules["sentence_transformers"] = st_mock

from src.reranker import cross_encoder_rerank, mmr_select, cosine_sim


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit_vec(values):
    """Normalise a list to a unit vector."""
    n = math.sqrt(sum(v * v for v in values)) or 1.0
    return [v / n for v in values]


def _make_candidates(n=5, dim=8):
    """Create n candidate dicts with distinct embeddings."""
    import random
    cands = []
    for i in range(n):
        rng = random.Random(i * 42)
        vec = _unit_vec([rng.gauss(0, 1) for _ in range(dim)])
        cands.append({
            "text": f"Candidate document {i} about topic {i % 3}",
            "source": f"doc{i}.txt",
            "chunk_index": i,
            "score": 1.0 / (i + 1),
            "embedding": vec,
        })
    return cands


# ---------------------------------------------------------------------------
# cosine_sim
# ---------------------------------------------------------------------------

class TestCosineSim:
    def test_identical_vectors(self):
        v = _unit_vec([1, 2, 3, 4])
        assert abs(cosine_sim(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        a = [1, 0, 0]
        b = [0, 1, 0]
        assert abs(cosine_sim(a, b)) < 1e-6

    def test_opposite_vectors(self):
        v = _unit_vec([1, 2, 3])
        neg_v = [-x for x in v]
        assert cosine_sim(v, neg_v) < -0.99

    def test_symmetry(self):
        a = _unit_vec([1, 2, 3])
        b = _unit_vec([4, 5, 6])
        assert abs(cosine_sim(a, b) - cosine_sim(b, a)) < 1e-9


# ---------------------------------------------------------------------------
# cross_encoder_rerank
# ---------------------------------------------------------------------------

class TestCrossEncoderRerank:
    def test_returns_list(self):
        cands = _make_candidates(5)
        result = cross_encoder_rerank("query text", cands, top_n=3)
        assert isinstance(result, list)

    def test_top_n_respected(self):
        cands = _make_candidates(5)
        result = cross_encoder_rerank("query", cands, top_n=3)
        assert len(result) <= 3

    def test_rerank_score_added(self):
        cands = _make_candidates(4)
        result = cross_encoder_rerank("test query", cands, top_n=4)
        for r in result:
            assert "rerank_score" in r
            assert isinstance(r["rerank_score"], float)

    def test_scores_descending(self):
        cands = _make_candidates(5)
        result = cross_encoder_rerank("deep learning", cands, top_n=5)
        scores = [r["rerank_score"] for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_empty_candidates(self):
        result = cross_encoder_rerank("query", [], top_n=3)
        assert result == []

    def test_custom_embed_fn(self):
        """Custom embed_fn should be called for query and each doc."""
        cands = _make_candidates(3)
        call_count = [0]

        def my_embed(text):
            call_count[0] += 1
            return _unit_vec([1.0] * 8)

        cross_encoder_rerank("question", cands, top_n=2, embed_fn=my_embed)
        # Should be called: 1 (query) + 0 (docs already have embeddings)
        assert call_count[0] >= 1

    def test_metadata_preserved(self):
        """Original doc fields should survive reranking."""
        cands = _make_candidates(3)
        result = cross_encoder_rerank("query", cands, top_n=3)
        for r in result:
            assert "source" in r
            assert "chunk_index" in r

    def test_docs_without_embeddings(self):
        """Docs without embedding field should still be scored."""
        cands = [
            {"text": "hello world", "source": "a", "chunk_index": 0},
            {"text": "goodbye world", "source": "b", "chunk_index": 1},
        ]
        result = cross_encoder_rerank("hello", cands, top_n=2)
        assert len(result) == 2
        for r in result:
            assert "rerank_score" in r

    def test_single_candidate(self):
        cands = _make_candidates(1)
        result = cross_encoder_rerank("anything", cands, top_n=3)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# mmr_select
# ---------------------------------------------------------------------------

class TestMMRSelect:
    def test_returns_list(self):
        cands = _make_candidates(5)
        q_emb = _unit_vec([1.0] * 8)
        result = mmr_select(cands, q_emb, top_n=3)
        assert isinstance(result, list)

    def test_top_n_respected(self):
        cands = _make_candidates(5)
        q_emb = _unit_vec([1.0] * 8)
        result = mmr_select(cands, q_emb, top_n=3)
        assert len(result) == 3

    def test_fewer_than_top_n(self):
        cands = _make_candidates(2)
        q_emb = _unit_vec([1.0] * 8)
        result = mmr_select(cands, q_emb, top_n=5)
        assert len(result) == 2

    def test_no_duplicates(self):
        cands = _make_candidates(6)
        q_emb = _unit_vec([1.0] * 8)
        result = mmr_select(cands, q_emb, top_n=4)
        chunk_ids = [r["chunk_index"] for r in result]
        assert len(chunk_ids) == len(set(chunk_ids))

    def test_empty_candidates(self):
        q_emb = _unit_vec([1.0] * 8)
        result = mmr_select([], q_emb, top_n=3)
        assert result == []

    def test_diversity_lambda_0(self):
        """lambda_=0 → pure diversity; should avoid redundant docs."""
        import random
        # Two clusters: cluster A ~ [1,0,...] and cluster B ~ [0,1,...]
        dim = 8
        cluster_a = [_unit_vec([1, 0] + [0] * (dim - 2))] * 4
        cluster_b = [_unit_vec([0, 1] + [0] * (dim - 2))]
        cands = (
            [{"text": f"A{i}", "source": "a", "chunk_index": i, "embedding": e}
             for i, e in enumerate(cluster_a)]
            + [{"text": "B0", "source": "b", "chunk_index": 10, "embedding": cluster_b[0]}]
        )
        q_emb = _unit_vec([1, 0] + [0] * (dim - 2))
        result = mmr_select(cands, q_emb, top_n=3, lambda_=0.0)
        # With lambda=0, after first pick from cluster A, MMR should pick cluster B next
        chunk_ids = [r["chunk_index"] for r in result]
        assert 10 in chunk_ids  # cluster B doc should be selected for diversity

    def test_relevance_lambda_1(self):
        """lambda_=1 → pure relevance; should pick highest cosine sim docs."""
        dim = 8
        q_emb = _unit_vec([1.0] + [0.0] * (dim - 1))
        # Create docs with explicit similarity to query
        cands = [
            {"text": "best", "source": "a", "chunk_index": 0, "embedding": _unit_vec([1.0] + [0.0] * (dim - 1))},
            {"text": "worst", "source": "b", "chunk_index": 1, "embedding": _unit_vec([0.0, 1.0] + [0.0] * (dim - 2))},
            {"text": "middle", "source": "c", "chunk_index": 2, "embedding": _unit_vec([0.7, 0.7] + [0.0] * (dim - 2))},
        ]
        result = mmr_select(cands, q_emb, top_n=1, lambda_=1.0)
        assert result[0]["chunk_index"] == 0  # most relevant to query

    def test_docs_without_embeddings(self):
        """mmr_select should compute embeddings on the fly if not present."""
        cands = [
            {"text": "hello world", "source": "a", "chunk_index": 0},
            {"text": "machine learning rocks", "source": "b", "chunk_index": 1},
        ]
        q_emb = _unit_vec([0.5] * 8)
        result = mmr_select(cands, q_emb, top_n=2)
        assert len(result) == 2
