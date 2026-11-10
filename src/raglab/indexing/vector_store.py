"""ChromaDB persistent vector store interface.

Manages collection lifecycle, idempotent chunk insertion, and vector similarity
search with metadata filtering and score attribution.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from pydantic import BaseModel
from raglab.ingestion.chunker import Chunk


class SearchResult(BaseModel):
    """Represents a retrieved chunk with similarity score and metadata.

    Attributes:
        chunk: The underlying retrieved text chunk.
        score: Relevance or similarity score (e.g., cosine similarity).
        metadata: Associated document and chunk metadata.
    """
    chunk: Chunk
    score: float
    metadata: Dict[str, Any]


class VectorStore:
    """Manages persistent vector collections using ChromaDB."""

    def __init__(self, collection_name: str, persist_directory: str) -> None:
        """Initialize ChromaDB client and retrieve or create collection.

        Args:
            collection_name: Target collection identifier.
            persist_directory: Filesystem path to persist vector store data.
        """
        pass

    def add_chunks(self, chunks: List[Chunk], embeddings: np.ndarray) -> None:
        """Idempotently add text chunks and their embeddings into the store.

        Args:
            chunks: List of document chunks to insert.
            embeddings: Corresponding embedding vectors.
        """
        raise NotImplementedError

    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Perform similarity search for top-k matching chunks.

        Args:
            query_embedding: Query embedding vector.
            top_k: Number of nearest neighbors to retrieve.
            filters: Optional metadata filtering criteria.

        Returns:
            List[SearchResult]: Nearest chunks sorted by similarity score.
        """
        raise NotImplementedError
