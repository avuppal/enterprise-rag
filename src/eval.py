"""
RAG evaluation framework.

Retrieval metrics:
    recall_at_k          — fraction of relevant docs retrieved in top-k
    mean_reciprocal_rank — MRR over a result list
    ndcg_at_k            — Normalised Discounted Cumulative Gain (binary relevance)

Answer quality metrics:
    rouge_l_score        — longest common subsequence F1 (ROUGE-L)
    faithfulness_score   — token-overlap between answer and retrieved context

Aggregate:
    run_eval             — evaluate a RAGPipeline over an entire dataset
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Tokenisation helper (shared)
# ---------------------------------------------------------------------------

def _tokens(text: str) -> List[str]:
    """Lowercase word tokens, punctuation removed."""
    return re.findall(r"[a-z0-9]+", text.lower())


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

def recall_at_k(
    relevant_ids: Sequence[str],
    retrieved_ids: Sequence[str],
    k: int,
) -> float:
    """
    Recall@K: fraction of relevant documents found in the top-k retrieved.

    Parameters
    ----------
    relevant_ids:  Ground-truth relevant document/chunk IDs.
    retrieved_ids: Ranked list of retrieved IDs (order matters for k cutoff).
    k:             Cutoff rank.

    Returns
    -------
    float in [0, 1].  Returns 0.0 if relevant_ids is empty.
    """
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    hits = sum(1 for rid in relevant_ids if rid in top_k)
    return hits / len(relevant_ids)


def mean_reciprocal_rank(
    relevant_ids: Sequence[str],
    retrieved_ids: Sequence[str],
) -> float:
    """
    Mean Reciprocal Rank (MRR) for a single query.

    The reciprocal rank is 1/rank of the *first* relevant document in the
    retrieved list.  Returns 0.0 if no relevant document appears.

    Parameters
    ----------
    relevant_ids:  Ground-truth relevant IDs.
    retrieved_ids: Ranked retrieved list.

    Returns
    -------
    float in [0, 1].
    """
    rel_set = set(relevant_ids)
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in rel_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    relevant_ids: Sequence[str],
    retrieved_ids: Sequence[str],
    k: int,
) -> float:
    """
    Normalised Discounted Cumulative Gain at K (binary relevance).

    DCG@K  = Σ_{i=1}^{K} rel_i / log2(i+1)
    IDCG@K = Σ_{i=1}^{min(|R|,K)} 1 / log2(i+1)   (ideal ordering)
    NDCG@K = DCG@K / IDCG@K

    Parameters
    ----------
    relevant_ids:  Ground-truth relevant IDs.
    retrieved_ids: Ranked retrieved list.
    k:             Cutoff rank.

    Returns
    -------
    float in [0, 1].  Returns 0.0 if relevant_ids is empty.
    """
    if not relevant_ids:
        return 0.0

    rel_set = set(relevant_ids)
    top_k = list(retrieved_ids[:k])

    dcg = sum(
        1.0 / math.log2(i + 2)  # i is 0-indexed → rank = i+1, denominator log2(rank+1)
        for i, rid in enumerate(top_k)
        if rid in rel_set
    )

    ideal_hits = min(len(rel_set), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Answer quality metrics
# ---------------------------------------------------------------------------

def _lcs_length(a: List[str], b: List[str]) -> int:
    """Compute length of longest common subsequence via DP."""
    m, n = len(a), len(b)
    # Space-optimised: only keep two rows
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


def rouge_l_score(reference: str, hypothesis: str) -> float:
    """
    ROUGE-L F1 score using the longest common subsequence.

    Parameters
    ----------
    reference:  Ground-truth answer string.
    hypothesis: Generated answer string.

    Returns
    -------
    float in [0, 1].
    """
    ref_toks = _tokens(reference)
    hyp_toks = _tokens(hypothesis)

    if not ref_toks or not hyp_toks:
        return 0.0

    lcs = _lcs_length(ref_toks, hyp_toks)
    precision = lcs / len(hyp_toks)
    recall = lcs / len(ref_toks)

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def faithfulness_score(answer: str, context_chunks: Sequence[Dict[str, Any]]) -> float:
    """
    Estimate faithfulness as the fraction of answer tokens that appear in context.

    This is a lightweight heuristic (token overlap); a production system would
    use an NLI model or LLM-as-judge.

    Parameters
    ----------
    answer:         Generated answer string.
    context_chunks: List of chunk dicts with a ``text`` key.

    Returns
    -------
    float in [0, 1].
    """
    answer_toks = set(_tokens(answer))
    if not answer_toks:
        return 0.0

    context_text = " ".join(c.get("text", "") for c in context_chunks)
    context_toks = set(_tokens(context_text))

    overlap = answer_toks & context_toks
    return len(overlap) / len(answer_toks)


# ---------------------------------------------------------------------------
# Aggregate evaluation
# ---------------------------------------------------------------------------

def run_eval(
    pipeline: Any,
    eval_dataset: List[Dict[str, Any]],
    top_k: int = 10,
    rerank_top_n: int = 3,
    k_recall: int = 5,
    k_ndcg: int = 5,
) -> Dict[str, float]:
    """
    Evaluate a RAGPipeline over an evaluation dataset.

    Dataset format (list of dicts):
        {
            "question":       str,
            "relevant_ids":   List[str],   # ground-truth chunk / doc IDs
            "reference_answer": str,        # ground-truth answer (optional)
        }

    Returns
    -------
    dict with aggregated metrics:
        recall_at_k, mrr, ndcg_at_k, faithfulness, rouge_l (if ref available)
    """
    if not eval_dataset:
        return {}

    recalls, mrrs, ndcgs, faithfulnesses, rouge_ls = [], [], [], [], []

    for item in eval_dataset:
        question = item["question"]
        relevant_ids = item.get("relevant_ids", [])
        reference = item.get("reference_answer", "")

        result = pipeline.query(question, top_k=top_k, rerank_top_n=rerank_top_n)
        reranked = result.get("reranked_chunks", [])
        retrieved = result.get("retrieved_chunks", [])

        # Build retrieved IDs from chunk metadata
        retrieved_ids = [
            f"{c.get('source', '')}::{c.get('chunk_index', i)}"
            for i, c in enumerate(retrieved)
        ]

        recalls.append(recall_at_k(relevant_ids, retrieved_ids, k=k_recall))
        mrrs.append(mean_reciprocal_rank(relevant_ids, retrieved_ids))
        ndcgs.append(ndcg_at_k(relevant_ids, retrieved_ids, k=k_ndcg))
        faithfulnesses.append(faithfulness_score(result["answer"], reranked))

        if reference:
            rouge_ls.append(rouge_l_score(reference, result["answer"]))

    metrics: Dict[str, float] = {
        "recall_at_k": sum(recalls) / len(recalls),
        "mrr": sum(mrrs) / len(mrrs),
        "ndcg_at_k": sum(ndcgs) / len(ndcgs),
        "faithfulness": sum(faithfulnesses) / len(faithfulnesses),
        "num_queries": float(len(eval_dataset)),
    }
    if rouge_ls:
        metrics["rouge_l"] = sum(rouge_ls) / len(rouge_ls)

    return metrics
