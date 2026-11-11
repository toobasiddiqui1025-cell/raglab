"""Sparse lexical BM25 index module.

Constructs, serializes, and queries keyword-based BM25 indices using rank_bm25
to support exact keyword matching and lexical search capabilities.
"""

import logging
import pickle
import re
from pathlib import Path
from typing import List, Optional, Tuple

from rank_bm25 import BM25Okapi

from raglab.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)

# Simple tokenizer: lowercase, split on non-alphanumeric characters
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    """Lowercase and tokenize text into word tokens.

    Args:
        text: Raw input string.

    Returns:
        List of lowercase alphanumeric tokens.
    """
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """Manages inverted index and BM25 scoring across document chunks."""

    def __init__(self) -> None:
        """Initialize an empty BM25 index."""
        self._chunks: List[Chunk] = []
        self._bm25: Optional[BM25Okapi] = None

    @property
    def is_built(self) -> bool:
        """Return True if the index has been built from at least one chunk."""
        return self._bm25 is not None

    def build(self, chunks: List[Chunk]) -> None:
        """Tokenize chunks and build the BM25 inverted index.

        Args:
            chunks: List of document chunks to index.

        Raises:
            ValueError: If chunks is empty.
        """
        if not chunks:
            raise ValueError("Cannot build BM25 index from an empty chunk list.")

        self._chunks = list(chunks)
        tokenized_corpus = [_tokenize(c.text) for c in self._chunks]
        self._bm25 = BM25Okapi(tokenized_corpus)
        logger.info("BM25 index built over %d chunks.", len(self._chunks))

    def query(self, query_text: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        """Retrieve top-k chunks scoring highest on BM25 for given query text.

        Args:
            query_text: Raw query string.
            top_k: Number of highest-ranking chunks to retrieve.

        Returns:
            List[Tuple[Chunk, float]]: Ranked list of (chunk, bm25_score) tuples,
            sorted by descending score.

        Raises:
            RuntimeError: If the index has not been built yet.
        """
        if not self.is_built:
            raise RuntimeError("BM25 index has not been built. Call build() first.")

        query_tokens = _tokenize(query_text)
        if not query_tokens:
            logger.warning("BM25 query produced no tokens after tokenization.")
            return []

        scores = self._bm25.get_scores(query_tokens)  # type: ignore[union-attr]
        top_k_clamped = min(top_k, len(self._chunks))

        # argsort descending
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        results: List[Tuple[Chunk, float]] = []
        for idx in ranked_indices[:top_k_clamped]:
            results.append((self._chunks[idx], float(scores[idx])))

        return results

    def save(self, file_path: Path) -> None:
        """Serialize index state to disk using pickle.

        Args:
            file_path: Destination path for saved index.

        Raises:
            RuntimeError: If the index has not been built yet.
        """
        if not self.is_built:
            raise RuntimeError("Cannot save an unbuilt index. Call build() first.")

        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump({"chunks": self._chunks, "bm25": self._bm25}, f)
        logger.info("BM25 index saved to '%s'.", file_path)

    def load(self, file_path: Path) -> None:
        """Load serialized BM25 index from disk.

        Args:
            file_path: Source path of saved index.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        file_path = Path(file_path)
        if not file_path.is_file():
            raise FileNotFoundError(f"BM25 index file not found: {file_path}")

        with open(file_path, "rb") as f:
            data = pickle.load(f)

        self._chunks = data["chunks"]
        self._bm25 = data["bm25"]
        logger.info(
            "BM25 index loaded from '%s'. %d chunks in index.", file_path, len(self._chunks)
        )
