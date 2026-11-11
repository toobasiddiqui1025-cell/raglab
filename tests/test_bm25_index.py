"""Tests for the BM25 sparse index (indexing/bm25_index.py)."""

import pickle
import tempfile
from pathlib import Path

import pytest

from raglab.indexing.bm25_index import BM25Index, _tokenize
from raglab.ingestion.chunker import Chunk


def _make_chunk(doc_id: str, page: int, idx: int, text: str) -> Chunk:
    return Chunk(
        doc_id=doc_id,
        page_number=page,
        chunk_id=f"{doc_id}_p{page}_c{idx}",
        text=text,
        token_count=len(text.split()),
    )


CHUNKS = [
    _make_chunk("doc1", 1, 0, "Attention mechanisms allow neural networks to focus on relevant parts."),
    _make_chunk("doc1", 2, 1, "BERT pre-trains deep bidirectional transformers on masked language modeling."),
    _make_chunk("doc2", 1, 0, "Retrieval augmented generation improves factual accuracy with external knowledge."),
    _make_chunk("doc2", 2, 1, "BM25 is a bag-of-words retrieval function used in information retrieval."),
    _make_chunk("doc3", 1, 0, "Reinforcement learning from human feedback improves language model alignment."),
]


class TestTokenize:
    def test_basic_tokenization(self) -> None:
        tokens = _tokenize("Hello, World! 123")
        assert tokens == ["hello", "world", "123"]

    def test_empty_string(self) -> None:
        assert _tokenize("") == []

    def test_special_chars_stripped(self) -> None:
        tokens = _tokenize("pre-trained: transformers.")
        assert "pre" in tokens and "trained" in tokens and "transformers" in tokens


class TestBM25IndexBuild:
    def test_build_succeeds(self) -> None:
        index = BM25Index()
        index.build(CHUNKS)
        assert index.is_built

    def test_build_empty_raises(self) -> None:
        index = BM25Index()
        with pytest.raises(ValueError, match="empty"):
            index.build([])

    def test_unbuilt_index_not_ready(self) -> None:
        index = BM25Index()
        assert not index.is_built


class TestBM25IndexQuery:
    @pytest.fixture
    def built_index(self) -> BM25Index:
        index = BM25Index()
        index.build(CHUNKS)
        return index

    def test_query_returns_tuples(self, built_index: BM25Index) -> None:
        results = built_index.query("attention transformer", top_k=3)
        assert len(results) == 3
        for chunk, score in results:
            assert isinstance(chunk, Chunk)
            assert isinstance(score, float)

    def test_top_result_relevance_retrieval(self, built_index: BM25Index) -> None:
        """'retrieval' query should rank the BM25/retrieval chunk highest."""
        results = built_index.query("retrieval BM25 information", top_k=1)
        assert results[0][0].doc_id == "doc2"

    def test_scores_sorted_descending(self, built_index: BM25Index) -> None:
        results = built_index.query("language model pre-training", top_k=5)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_respected(self, built_index: BM25Index) -> None:
        results = built_index.query("neural network", top_k=2)
        assert len(results) == 2

    def test_top_k_clamped_to_corpus_size(self, built_index: BM25Index) -> None:
        results = built_index.query("attention", top_k=100)
        assert len(results) == len(CHUNKS)

    def test_empty_query_tokens_returns_empty(self, built_index: BM25Index, caplog: pytest.LogCaptureFixture) -> None:
        import logging
        with caplog.at_level(logging.WARNING, logger="raglab.indexing.bm25_index"):
            results = built_index.query("!!! @@@", top_k=3)
        assert results == []

    def test_query_before_build_raises(self) -> None:
        index = BM25Index()
        with pytest.raises(RuntimeError, match="not been built"):
            index.query("test query")


class TestBM25IndexPersistence:
    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        index = BM25Index()
        index.build(CHUNKS)

        save_path = tmp_path / "bm25.pkl"
        index.save(save_path)
        assert save_path.exists()

        loaded = BM25Index()
        loaded.load(save_path)
        assert loaded.is_built

        # Results should be identical after round-trip
        original_results = index.query("retrieval augmented", top_k=3)
        loaded_results = loaded.query("retrieval augmented", top_k=3)
        orig_ids = [c.chunk_id for c, _ in original_results]
        loaded_ids = [c.chunk_id for c, _ in loaded_results]
        assert orig_ids == loaded_ids

    def test_save_unbuilt_raises(self, tmp_path: Path) -> None:
        index = BM25Index()
        with pytest.raises(RuntimeError, match="unbuilt"):
            index.save(tmp_path / "bm25.pkl")

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        index = BM25Index()
        with pytest.raises(FileNotFoundError):
            index.load(tmp_path / "nonexistent.pkl")

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        index = BM25Index()
        index.build(CHUNKS)
        nested_path = tmp_path / "nested" / "dir" / "bm25.pkl"
        index.save(nested_path)
        assert nested_path.exists()
