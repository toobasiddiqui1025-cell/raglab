"""ChromaDB persistent vector store interface.

Manages collection lifecycle, idempotent chunk insertion, and vector similarity
search with metadata filtering and score attribution.
"""

import logging
from typing import Any, Dict, List, Optional

import chromadb
import numpy as np
from chromadb.config import Settings
from pydantic import BaseModel

from raglab.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)


class SearchResult(BaseModel):
    """Represents a retrieved chunk with similarity score and metadata.

    Attributes:
        chunk: The underlying retrieved text chunk.
        score: Relevance or similarity score (cosine similarity, 0–1 for normalized vectors).
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
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        logger.info(
            "Connecting to ChromaDB at '%s', collection '%s'",
            persist_directory,
            collection_name,
        )
        self._client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "Collection '%s' ready. Current count: %d",
            collection_name,
            self._collection.count(),
        )

    @property
    def count(self) -> int:
        """Return the number of embeddings currently stored in the collection."""
        return self._collection.count()

    def add_chunks(self, chunks: List[Chunk], embeddings: np.ndarray) -> None:
        """Idempotently add text chunks and their embeddings into the store.

        Chunks whose chunk_id already exists in the collection are skipped.

        Args:
            chunks: List of document chunks to insert.
            embeddings: Corresponding embedding vectors with shape (len(chunks), dim).

        Raises:
            ValueError: If len(chunks) != len(embeddings).
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Number of chunks ({len(chunks)}) must match number of embeddings "
                f"({len(embeddings)})."
            )

        if not chunks:
            logger.warning("add_chunks called with an empty list — nothing to insert.")
            return

        # Filter out chunk_ids that are already in the collection
        existing_ids = set(self._collection.get(ids=[c.chunk_id for c in chunks])["ids"])
        new_chunks = [c for c in chunks if c.chunk_id not in existing_ids]
        new_embeddings = np.array(
            [embeddings[i] for i, c in enumerate(chunks) if c.chunk_id not in existing_ids]
        )

        if not new_chunks:
            logger.info("All %d chunks already in collection — skipping.", len(chunks))
            return

        self._collection.add(
            ids=[c.chunk_id for c in new_chunks],
            embeddings=new_embeddings.tolist(),
            documents=[c.text for c in new_chunks],
            metadatas=[
                {
                    "doc_id": c.doc_id,
                    "page_number": c.page_number,
                    "chunk_id": c.chunk_id,
                    "token_count": c.token_count,
                }
                for c in new_chunks
            ],
        )
        logger.info(
            "Inserted %d new chunks (skipped %d duplicates).",
            len(new_chunks),
            len(chunks) - len(new_chunks),
        )

    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Perform similarity search for top-k matching chunks.

        Args:
            query_embedding: 1D query embedding vector.
            top_k: Number of nearest neighbors to retrieve.
            filters: Optional ChromaDB ``where`` clause for metadata filtering.

        Returns:
            List[SearchResult]: Nearest chunks sorted by descending similarity score.

        Raises:
            ValueError: If the collection is empty.
        """
        if self._collection.count() == 0:
            raise ValueError("Cannot query an empty collection. Add chunks first.")

        raw = self._collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, self._collection.count()),
            where=filters,
            include=["documents", "metadatas", "distances"],
        )

        results: List[SearchResult] = []
        for doc, meta, distance in zip(
            raw["documents"][0],
            raw["metadatas"][0],
            raw["distances"][0],
        ):
            # ChromaDB cosine distance = 1 - cosine_similarity → convert back
            score = 1.0 - float(distance)
            chunk = Chunk(
                doc_id=meta["doc_id"],
                page_number=int(meta["page_number"]),
                chunk_id=meta["chunk_id"],
                text=doc,
                token_count=int(meta["token_count"]),
            )
            results.append(SearchResult(chunk=chunk, score=score, metadata=meta))

        return results

    def delete_collection(self) -> None:
        """Delete the entire collection from the persistent store.

        Useful for resetting experiments or clearing stale data.
        """
        self._client.delete_collection(self.collection_name)
        logger.info("Deleted collection '%s'.", self.collection_name)
