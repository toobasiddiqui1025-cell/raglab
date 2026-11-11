"""Tests for the ChromaDB vector store (indexing/vector_store.py)."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from raglab.indexing.embedder import Embedder
from raglab.indexing.vector_store import SearchResult, VectorStore
from raglab.ingestion.chunker import Chunk

MODEL = "BAAI/bge-small-en-v1.5"


def _make_chunk(doc_id: str, page: int, idx: int, text: str) -> Chunk:
    return Chunk(
        doc_id=doc_id,
        page_number=page,
        chunk_id=f"{doc_id}_p{page}_c{idx}",
        text=text,
        token_count=len(text.split()),
    )


SAMPLE_CHUNKS = [
    _make_chunk("paper1", 1, 0, "Attention mechanisms allow models to focus on relevant tokens."),
    _make_chunk("paper1", 2, 1, "BERT uses bidirectional transformer encoders pre-trained on large corpora."),
    _make_chunk("paper2", 1, 0, "Retrieval augmented generation combines dense retrieval with language models."),
    _make_chunk("paper2", 2, 1, "BM25 is a probabilistic term-weighting ranking function."),
    _make_chunk("paper3", 1, 0, "Reinforcement learning optimizes agents via reward signals."),
]


@pytest.fixture(scope="module")
def embedder() -> Embedder:
    return Embedder(model_name=MODEL, batch_size=8)


@pytest.fixture
def store(tmp_path: Path) -> VectorStore:
    """Fresh VectorStore backed by a temporary directory."""
    return VectorStore(collection_name="test_collection", persist_directory=str(tmp_path))


@pytest.fixture
def populated_store(store: VectorStore, embedder: Embedder) -> VectorStore:
    texts = [c.text for c in SAMPLE_CHUNKS]
    embeddings = embedder.embed_texts(texts)
    store.add_chunks(SAMPLE_CHUNKS, embeddings)
    return store


class TestVectorStoreInit:
    def test_empty_on_creation(self, store: VectorStore) -> None:
        assert store.count == 0


class TestAddChunks:
    def test_add_chunks_increases_count(self, store: VectorStore, embedder: Embedder) -> None:
        texts = [c.text for c in SAMPLE_CHUNKS]
        embeddings = embedder.embed_texts(texts)
        store.add_chunks(SAMPLE_CHUNKS, embeddings)
        assert store.count == len(SAMPLE_CHUNKS)

    def test_idempotent_insert(self, store: VectorStore, embedder: Embedder) -> None:
        """Inserting the same chunks twice must not duplicate them."""
        texts = [c.text for c in SAMPLE_CHUNKS]
        embeddings = embedder.embed_texts(texts)
        store.add_chunks(SAMPLE_CHUNKS, embeddings)
        store.add_chunks(SAMPLE_CHUNKS, embeddings)
        assert store.count == len(SAMPLE_CHUNKS)

    def test_mismatched_lengths_raises(self, store: VectorStore) -> None:
        dummy_emb = np.random.rand(3, 384).astype(np.float32)
        with pytest.raises(ValueError, match="must match"):
            store.add_chunks(SAMPLE_CHUNKS, dummy_emb)

    def test_empty_chunks_logs_warning(self, store: VectorStore, caplog: pytest.LogCaptureFixture) -> None:
        import logging
        with caplog.at_level(logging.WARNING, logger="raglab.indexing.vector_store"):
            store.add_chunks([], np.empty((0, 384)))
        assert any("empty" in r.message.lower() for r in caplog.records)


class TestQuery:
    def test_returns_search_results(self, populated_store: VectorStore, embedder: Embedder) -> None:
        query_emb = embedder.embed_query("attention transformer model")
        results = populated_store.query(query_emb, top_k=3)
        assert len(results) == 3
        assert all(isinstance(r, SearchResult) for r in results)

    def test_scores_between_0_and_1(self, populated_store: VectorStore, embedder: Embedder) -> None:
        query_emb = embedder.embed_query("retrieval language model")
        results = populated_store.query(query_emb, top_k=5)
        for r in results:
            assert 0.0 <= r.score <= 1.0, f"Score out of range: {r.score}"

    def test_results_sorted_descending(self, populated_store: VectorStore, embedder: Embedder) -> None:
        query_emb = embedder.embed_query("BERT pre-training")
        results = populated_store.query(query_emb, top_k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_respected(self, populated_store: VectorStore, embedder: Embedder) -> None:
        query_emb = embedder.embed_query("machine learning")
        results = populated_store.query(query_emb, top_k=2)
        assert len(results) == 2

    def test_top_result_relevance(self, populated_store: VectorStore, embedder: Embedder) -> None:
        """The top result for a retrieval query should come from paper2."""
        query_emb = embedder.embed_query("retrieval augmented generation RAG")
        results = populated_store.query(query_emb, top_k=1)
        assert results[0].chunk.doc_id == "paper2"

    def test_query_empty_store_raises(self, store: VectorStore, embedder: Embedder) -> None:
        query_emb = embedder.embed_query("empty store query")
        with pytest.raises(ValueError, match="empty"):
            store.query(query_emb)

    def test_chunk_metadata_preserved(self, populated_store: VectorStore, embedder: Embedder) -> None:
        query_emb = embedder.embed_query("attention self-attention")
        results = populated_store.query(query_emb, top_k=1)
        r = results[0]
        assert r.chunk.doc_id == r.metadata["doc_id"]
        assert r.chunk.chunk_id == r.metadata["chunk_id"]


class TestDeleteCollection:
    def test_delete_removes_collection(self, tmp_path: Path) -> None:
        store = VectorStore(collection_name="to_delete", persist_directory=str(tmp_path / "del"))
        store.delete_collection()
        # Re-creating the same collection should start at 0
        store2 = VectorStore(collection_name="to_delete", persist_directory=str(tmp_path / "del"))
        assert store2.count == 0
