"""Sparse lexical retrieval module using BM25.

Provides keyword-based search over indexed chunks to complement semantic retrieval
on precise terms, acronyms, author names, and domain-specific terminology.
"""

from typing import List
from raglab.indexing.bm25_index import BM25Index
from raglab.indexing.vector_store import SearchResult


class SparseRetriever:
    """Retrieves chunks based on BM25 lexical keyword scores."""

    def __init__(self, bm25_index: BM25Index) -> None:
        """Initialize sparse retriever with a populated BM25 index.

        Args:
            bm25_index: Built or loaded BM25Index instance.
        """
        pass

    def retrieve(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Retrieve top-k chunks ranking highest by BM25 score.

        Args:
            query: Query text string.
            top_k: Number of chunks to retrieve.

        Returns:
            List[SearchResult]: Retrieved chunks with BM25 relevance scores.
        """
        raise NotImplementedError
