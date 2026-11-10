"""Sparse lexical BM25 index module.

Constructs, serializes, and queries keyword-based BM25 indices using rank_bm25
to support exact keyword matching and lexical search capabilities.
"""

from pathlib import Path
from typing import List, Tuple
from raglab.ingestion.chunker import Chunk


class BM25Index:
    """Manages inverted index and BM25 scoring across document chunks."""

    def __init__(self) -> None:
        """Initialize an empty BM25 index."""
        pass

    def build(self, chunks: List[Chunk]) -> None:
        """Tokenize chunks and build the BM25 inverted index.

        Args:
            chunks: List of document chunks to index.
        """
        raise NotImplementedError

    def query(self, query_text: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        """Retrieve top-k chunks scoring highest on BM25 for given query text.

        Args:
            query_text: Raw query string.
            top_k: Number of highest-ranking chunks to retrieve.

        Returns:
            List[Tuple[Chunk, float]]: Ranked list of (chunk, bm25_score) tuples.
        """
        raise NotImplementedError

    def save(self, file_path: Path) -> None:
        """Serialize index state to disk.

        Args:
            file_path: Destination path for saved index.
        """
        raise NotImplementedError

    def load(self, file_path: Path) -> None:
        """Load serialized BM25 index from disk.

        Args:
            file_path: Source path of saved index.
        """
        raise NotImplementedError
