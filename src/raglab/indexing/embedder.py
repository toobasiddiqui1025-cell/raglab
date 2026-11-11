"""Dense embedding generation module using Sentence Transformers.

Generates normalized dense vector representations for text chunks and search
queries with batch processing and progress monitoring.
"""

import logging
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Embedder:
    """Generates dense vector embeddings using sentence-transformer models."""

    def __init__(self, model_name: str, batch_size: int = 32, normalize: bool = True) -> None:
        """Initialize the embedding model and runtime options.

        Args:
            model_name: HuggingFace sentence-transformers model identifier.
            batch_size: Batch size for batched inference.
            normalize: Whether to L2-normalize vectors for cosine similarity computation.
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize

        logger.info("Loading embedding model: %s", model_name)
        self._model = SentenceTransformer(model_name)
        logger.info("Embedding model loaded. Dimension: %d", self.embedding_dim)

    @property
    def embedding_dim(self) -> int:
        """Return the output embedding dimension of the loaded model."""
        # `get_sentence_embedding_dimension` was renamed in newer sentence-transformers versions
        get_dim = getattr(
            self._model,
            "get_embedding_dimension",
            getattr(self._model, "get_sentence_embedding_dimension", None),
        )
        return get_dim()

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of document texts.

        Args:
            texts: List of string chunks to embed.

        Returns:
            np.ndarray: 2D array of embeddings with shape (len(texts), embedding_dim).

        Raises:
            ValueError: If texts list is empty.
        """
        if not texts:
            raise ValueError("texts must be a non-empty list of strings.")

        logger.info("Embedding %d texts in batches of %d…", len(texts), self.batch_size)

        all_embeddings: List[np.ndarray] = []
        for start in tqdm(range(0, len(texts), self.batch_size), desc="Embedding", unit="batch"):
            batch = texts[start : start + self.batch_size]
            batch_embeddings = self._model.encode(
                batch,
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            all_embeddings.append(batch_embeddings)

        embeddings = np.vstack(all_embeddings)
        logger.info("Finished embedding. Output shape: %s", embeddings.shape)
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding vector for a single search query.

        Args:
            query: User search query string.

        Returns:
            np.ndarray: 1D normalized embedding vector.

        Raises:
            ValueError: If query is an empty string.
        """
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string.")

        embedding: np.ndarray = self._model.encode(
            query,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        # encode returns 1D for a single string — ensure consistent shape
        return embedding.flatten()
