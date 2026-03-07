"""
Tests for src/ingest.py — chunking correctness, overlap, metadata preservation,
and ingest_documents pipeline (ChromaDB mocked via sys.modules).
"""
import sys
import types
import math
import tempfile
import os
import pytest


# ---------------------------------------------------------------------------
# Mock ChromaDB before importing ingest
# ---------------------------------------------------------------------------

def _make_chroma_mock():
    chroma_mod = types.ModuleType("chromadb")

    class FakeCollection:
        def __init__(self):
            self.data = {}

        def upsert(self, ids, documents, embeddings, metadatas):
            for id_, doc, emb, meta in zip(ids, documents, embeddings, metadatas):
                self.data[id_] = {"doc": doc, "emb": emb, "meta": meta}

        def query(self, query_embeddings=None, query_texts=None, n_results=5, include=None):
            docs = list(self.data.values())[:n_results]
            return {
                "documents": [[d["doc"] for d in docs]],
                "metadatas": [[d["meta"] for d in docs]],
                "distances": [[0.1] * len(docs)],
            }

    class FakeClient:
        def get_or_create_collection(self, name, **kwargs):
            return FakeCollection()

    chroma_mod.Client = FakeClient
    chroma_mod.EphemeralClient = FakeClient
    return chroma_mod


# Inject mocks *before* importing ingest
if "chromadb" not in sys.modules:
    sys.modules["chromadb"] = _make_chroma_mock()

# Mock sentence_transformers to avoid heavy download
if "sentence_transformers" not in sys.modules:
    st_mock = types.ModuleType("sentence_transformers")

    class FakeSentenceTransformer:
        def __init__(self, model_name):
            self.dim = 32

        def encode(self, texts, show_progress_bar=False):
            import random
            results = []
            for _ in texts:
                vec = [random.gauss(0, 1) for _ in range(self.dim)]
                n = math.sqrt(sum(v * v for v in vec)) or 1.0
                results.append([v / n for v in vec])
            return results

    st_mock.SentenceTransformer = FakeSentenceTransformer
    sys.modules["sentence_transformers"] = st_mock

from src.ingest import chunk_text, embed_chunks, ingest_documents


# ---------------------------------------------------------------------------
# chunk_text
# ---------------------------------------------------------------------------

class TestChunkText:
    def test_basic_split(self):
        text = "a" * 100
        chunks = chunk_text(text, chunk_size=40, overlap=10)
        assert len(chunks) > 1
        for c in chunks:
            assert "text" in c
            assert "start_char" in c
            assert "end_char" in c
            assert "chunk_index" in c

    def test_chunk_size_respected(self):
        text = "x" * 200
        chunks = chunk_text(text, chunk_size=50, overlap=0)
        for c in chunks:
            assert len(c["text"]) <= 50

    def test_overlap_correctness(self):
        text = "abcdefghij"  # 10 chars
        chunks = chunk_text(text, chunk_size=6, overlap=2)
        # Second chunk should start at position chunk_size - overlap = 4
        assert chunks[0]["start_char"] == 0
        assert chunks[1]["start_char"] == 4

    def test_single_chunk_short_text(self):
        text = "Hello world"
        chunks = chunk_text(text, chunk_size=100, overlap=10)
        assert len(chunks) == 1
        assert chunks[0]["text"] == text
        assert chunks[0]["chunk_index"] == 0

    def test_chunk_indices_sequential(self):
        text = "z" * 300
        chunks = chunk_text(text, chunk_size=50, overlap=10)
        for i, c in enumerate(chunks):
            assert c["chunk_index"] == i

    def test_start_end_chars(self):
        text = "Hello world"
        chunks = chunk_text(text, chunk_size=5, overlap=0)
        assert chunks[0]["start_char"] == 0
        assert chunks[0]["end_char"] == 5
        assert chunks[1]["start_char"] == 5

    def test_no_data_loss(self):
        """Reassembling chunks with overlap=0 should reproduce the text."""
        text = "The quick brown fox jumps over the lazy dog"
        chunks = chunk_text(text, chunk_size=10, overlap=0)
        reassembled = "".join(c["text"] for c in chunks)
        assert reassembled == text

    def test_overlap_content(self):
        text = "0123456789"
        chunks = chunk_text(text, chunk_size=6, overlap=3)
        # chunk 0: "012345", chunk 1 starts at 3: "345678"
        overlap_content = chunks[0]["text"][-3:]
        assert chunks[1]["text"].startswith(overlap_content)

    def test_invalid_chunk_size(self):
        with pytest.raises(ValueError):
            chunk_text("hello", chunk_size=0)

    def test_invalid_overlap(self):
        with pytest.raises(ValueError):
            chunk_text("hello", chunk_size=5, overlap=5)

    def test_empty_text(self):
        chunks = chunk_text("", chunk_size=100, overlap=10)
        # Empty text produces no chunks or one empty chunk
        if chunks:
            assert chunks[0]["text"] == ""

    def test_exact_fit(self):
        """Text exactly fills one chunk."""
        text = "a" * 50
        chunks = chunk_text(text, chunk_size=50, overlap=0)
        assert len(chunks) == 1
        assert chunks[0]["text"] == text


# ---------------------------------------------------------------------------
# embed_chunks
# ---------------------------------------------------------------------------

class TestEmbedChunks:
    def _make_chunks(self, n=3):
        return [{"text": f"chunk {i}", "chunk_index": i} for i in range(n)]

    def test_embedding_added(self):
        chunks = self._make_chunks(3)
        result = embed_chunks(chunks, model_name="all-MiniLM-L6-v2")
        for c in result:
            assert "embedding" in c
            assert isinstance(c["embedding"], list)
            assert len(c["embedding"]) > 0

    def test_embedding_is_unit_vector(self):
        chunks = self._make_chunks(1)
        result = embed_chunks(chunks, model_name="all-MiniLM-L6-v2")
        emb = result[0]["embedding"]
        norm = math.sqrt(sum(v * v for v in emb))
        assert abs(norm - 1.0) < 1e-3

    def test_returns_same_list(self):
        chunks = self._make_chunks(2)
        result = embed_chunks(chunks)
        assert result is chunks  # in-place modification + return

    def test_metadata_preserved(self):
        chunks = [{"text": "hello", "chunk_index": 0, "source": "doc.txt"}]
        result = embed_chunks(chunks)
        assert result[0]["source"] == "doc.txt"


# ---------------------------------------------------------------------------
# ingest_documents
# ---------------------------------------------------------------------------

class TestIngestDocuments:
    def _fake_collection(self):
        sys.modules["chromadb"].Client().get_or_create_collection("test")
        return sys.modules["chromadb"].Client().get_or_create_collection("test")

    def test_txt_file_ingested(self):
        collection = self._fake_collection()
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
            f.write("Hello world. This is a test document.")
            path = f.name
        try:
            total = ingest_documents([path], collection, chunk_size=20, overlap=5)
            assert total > 0
        finally:
            os.unlink(path)

    def test_md_file_ingested(self):
        collection = self._fake_collection()
        with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False) as f:
            f.write("# Title\n\nSome markdown content here.")
            path = f.name
        try:
            total = ingest_documents([path], collection, chunk_size=50, overlap=5)
            assert total >= 1
        finally:
            os.unlink(path)

    def test_missing_file_raises(self):
        collection = self._fake_collection()
        with pytest.raises(FileNotFoundError):
            ingest_documents(["/nonexistent/path.txt"], collection)

    def test_multiple_files(self):
        collection = self._fake_collection()
        paths = []
        try:
            for i in range(3):
                with tempfile.NamedTemporaryFile(
                    suffix=".txt", mode="w", delete=False
                ) as f:
                    f.write(f"Document {i} content. " * 5)
                    paths.append(f.name)
            total = ingest_documents(paths, collection, chunk_size=30, overlap=5)
            assert total >= 3
        finally:
            for p in paths:
                os.unlink(p)


# ---------------------------------------------------------------------------
# late_chunk tests  (closes #1)
# ---------------------------------------------------------------------------

from src.ingest import late_chunk


class TestLateChunk:
    """Tests for the late-chunking strategy."""

    def test_output_structure(self):
        """Every chunk must have the required keys."""
        text = "Hello world. " * 50
        chunks = late_chunk(text, chunk_size=64, overlap=8)
        assert len(chunks) > 0
        required_keys = {"text", "start_char", "end_char", "chunk_index", "embedding", "strategy"}
        for chunk in chunks:
            assert required_keys.issubset(chunk.keys()), (
                f"Chunk missing keys: {required_keys - chunk.keys()}"
            )
            assert chunk["strategy"] == "late"

    def test_embedding_dimensionality(self):
        """All embeddings must have the same positive dimensionality."""
        text = "The quick brown fox jumps over the lazy dog. " * 20
        chunks = late_chunk(text, chunk_size=80, overlap=10)
        assert len(chunks) > 0
        dims = [len(c["embedding"]) for c in chunks]
        # All embeddings must share the same dimension
        assert len(set(dims)) == 1, f"Inconsistent embedding dims: {set(dims)}"
        assert dims[0] > 0

    def test_no_data_loss(self):
        """
        Every character of the original document must appear in at least one chunk.
        With overlap=0 the chunks concatenate to the exact original text.
        """
        text = "abcdefghijklmnopqrstuvwxyz" * 5  # 130 chars
        chunks = late_chunk(text, chunk_size=26, overlap=0)
        reconstructed = "".join(c["text"] for c in chunks)
        assert reconstructed == text, (
            "Reassembled text does not match original (data loss detected)"
        )
