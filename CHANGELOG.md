# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Planned
- Async retrieval for parallel dense + BM25 calls
- OpenTelemetry span export for distributed tracing
- LLM-as-judge faithfulness scorer (GPT-4 / Claude)
- HuggingFace Datasets integration for eval benchmark loading

---

## [1.0.0] — 2026-03-07

### Added
- **`src/ingest.py`** — Document ingestion pipeline
  - `chunk_text()` — character-level chunking with configurable overlap
  - `embed_chunks()` — sentence-transformers embeddings with CPU fallback
  - `ingest_documents()` — full pipeline: load → chunk → embed → upsert
  - PDF support via `pypdf`, plain text and Markdown natively
- **`src/retriever.py`** — Hybrid retrieval
  - `BM25Retriever` — pure-Python Okapi BM25 (no external library)
  - `DenseRetriever` — ChromaDB wrapper with pluggable embed function
  - `HybridRetriever` — Reciprocal Rank Fusion of dense + sparse
- **`src/reranker.py`** — Re-ranking strategies
  - `cross_encoder_rerank()` — CPU-safe cosine-sim mock; real CE is a drop-in
  - `mmr_select()` — Maximum Marginal Relevance for diversity-aware selection
  - `cosine_sim()` — standalone utility exposed for external use
- **`src/pipeline.py`** — End-to-end RAG pipeline
  - `RAGPipeline` — injectable retriever, reranker, and LLM callable
  - `query()` — returns answer, sources, chunks, and per-stage latency_ms
  - `batch_query()` — convenience wrapper for multi-query evaluation
- **`src/eval.py`** — Evaluation metrics
  - `recall_at_k()`, `mean_reciprocal_rank()`, `ndcg_at_k()`
  - `rouge_l_score()` — LCS-based ROUGE-L F1
  - `faithfulness_score()` — token-overlap heuristic
  - `run_eval()` — aggregate evaluation over a full dataset
- **`configs/rag_config.yaml`** — All tuneable parameters
- **`docs/architecture.md`** — Four Mermaid diagrams
- **35+ unit tests** — all CPU-only, external deps mocked
- Initial benchmark table in README (Dense · BM25 · Hybrid · Hybrid+CE)

---

## Links

- [Repository](https://github.com/avuppal/enterprise-rag)
- [Issues](https://github.com/avuppal/enterprise-rag/issues)
