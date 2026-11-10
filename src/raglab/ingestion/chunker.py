"""Document chunking and text splitting strategies.

Provides an extensible Chunker interface and implementations (FixedSizeChunker,
RecursiveChunker) to segment pages into retrieval chunks while tracking token counts,
page boundaries, and document metadata.
"""

from abc import ABC, abstractmethod
from typing import List
from pydantic import BaseModel
from raglab.ingestion.loader import Page


class Chunk(BaseModel):
    """Represents a discrete text chunk suitable for indexing and retrieval.

    Attributes:
        doc_id: Unique identifier of the originating document.
        page_number: Page number where the chunk content originated.
        chunk_id: Sequential index or unique identifier of the chunk within the document.
        text: Raw text content of the chunk.
        token_count: Number of tokens computed using a tokenizer.
    """


class Chunker(ABC):
    """Abstract base class defining the document chunking interface."""

    @abstractmethod
    def chunk(self, pages: List[Page]) -> List[Chunk]:
        """Split a collection of pages into retrieval chunks.

        Args:
            pages: List of extracted document pages.

        Returns:
            List[Chunk]: Processed chunks ready for indexing.
        """
        pass


class FixedSizeChunker(Chunker):
    """Chunks text into fixed-size token or character windows with configurable overlap."""

    def chunk(self, pages: List[Page]) -> List[Chunk]:
        """Split pages using fixed window size and overlap."""
        raise NotImplementedError


class RecursiveChunker(Chunker):
    """Splits text hierarchically across paragraph, sentence, and word boundaries."""

    def chunk(self, pages: List[Page]) -> List[Chunk]:
        """Split pages hierarchically respecting linguistic boundaries."""
        raise NotImplementedError
