"""Tests for the dense embedding module (indexing/embedder.py)."""

import numpy as np
import pytest

from raglab.indexing.embedder import Embedder

# Use a tiny model so tests run fast in CI
MODEL = "BAAI/bge-small-en-v1.5"


@pytest.fixture(scope="module")
def embedder() -> Embedder:
    """Shared Embedder instance for the test module."""
    return Embedder(model_name=MODEL, batch_size=8, normalize=True)


class TestEmbedderInit:
    def test_model_loads(self, embedder: Embedder) -> None:
        assert embedder is not None

    def test_embedding_dim_positive(self, embedder: Embedder) -> None:
        assert embedder.embedding_dim > 0


class TestEmbedTexts:
    def test_single_text(self, embedder: Embedder) -> None:
        result = embedder.embed_texts(["Hello world"])
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, embedder.embedding_dim)

    def test_multiple_texts(self, embedder: Embedder) -> None:
        texts = ["Attention is all you need.", "BERT: Pre-training deep bidirectional transformers."]
        result = embedder.embed_texts(texts)
        assert result.shape == (2, embedder.embedding_dim)

    def test_normalized_vectors(self, embedder: Embedder) -> None:
        result = embedder.embed_texts(["normalize this"])
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_empty_list_raises(self, embedder: Embedder) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            embedder.embed_texts([])

    def test_batching_produces_same_result(self, embedder: Embedder) -> None:
        texts = [f"sentence number {i}" for i in range(20)]
        result_small_batch = Embedder(model_name=MODEL, batch_size=4).embed_texts(texts)
        result_large_batch = Embedder(model_name=MODEL, batch_size=32).embed_texts(texts)
        np.testing.assert_allclose(result_small_batch, result_large_batch, atol=1e-5)


class TestEmbedQuery:
    def test_returns_1d_array(self, embedder: Embedder) -> None:
        result = embedder.embed_query("what is retrieval augmented generation?")
        assert isinstance(result, np.ndarray)
        assert result.ndim == 1
        assert result.shape[0] == embedder.embedding_dim

    def test_normalized(self, embedder: Embedder) -> None:
        result = embedder.embed_query("test query")
        norm = np.linalg.norm(result)
        assert abs(norm - 1.0) < 1e-5

    def test_empty_query_raises(self, embedder: Embedder) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            embedder.embed_query("")

    def test_whitespace_only_raises(self, embedder: Embedder) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            embedder.embed_query("   ")

    def test_query_embedding_similar_to_text_embedding(self, embedder: Embedder) -> None:
        """Query embedding for a text should be very similar to its text embedding."""
        text = "transformer architecture self-attention"
        query_emb = embedder.embed_query(text)
        text_emb = embedder.embed_texts([text])[0]
        cosine_sim = float(np.dot(query_emb, text_emb))
        assert cosine_sim > 0.95, f"Expected high similarity, got {cosine_sim:.4f}"
