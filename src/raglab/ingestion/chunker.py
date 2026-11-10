"""Document chunking and text splitting strategies.

Provides an extensible Chunker interface and implementations (FixedSizeChunker,
RecursiveChunker) to segment pages into retrieval chunks while tracking token counts,
page boundaries, and document metadata.
"""

from abc import ABC, abstractmethod
import logging
import re
from typing import List, Optional, Set
from pydantic import BaseModel, Field
import tiktoken
from raglab.config import IngestionConfig
from raglab.ingestion.loader import Page

logger = logging.getLogger(__name__)


class Chunk(BaseModel):
    """Represents a discrete text chunk suitable for indexing and retrieval.

    Attributes:
        doc_id: Unique identifier of the originating document.
        page_number: 1-indexed page number where the chunk content originated.
        chunk_id: Stable, unique identifier of the chunk (e.g. {doc_id}_p{page}_c{idx}).
        text: Raw text content of the chunk.
        token_count: Number of tokens computed using the configured tokenizer.
    """

    doc_id: str
    page_number: int
    chunk_id: str
    text: str
    token_count: int


class Piece:
    """Internal representation of an atomic text segment with its preceding separator."""

    def __init__(self, text: str, separator: str = "") -> None:
        self.text = text
        self.separator = separator


class Chunker(ABC):
    """Abstract base class defining the document chunking interface."""

    def __init__(
        self,
        chunk_size_tokens: int = 500,
        chunk_overlap_tokens: int = 50,
        encoding_name: str = "cl100k_base",
    ) -> None:
        """Initialize chunker with token limits and tokenizer encoding.

        Args:
            chunk_size_tokens: Maximum allowed tokens per chunk.
            chunk_overlap_tokens: Token overlap between consecutive chunks.
            encoding_name: Tiktoken encoding name (default: 'cl100k_base').

        Raises:
            ValueError: If chunk_size_tokens <= 0 or chunk_overlap_tokens >= chunk_size_tokens.
        """
        if chunk_size_tokens <= 0:
            raise ValueError(
                f"chunk_size_tokens must be positive, got {chunk_size_tokens}"
            )
        if chunk_overlap_tokens < 0:
            raise ValueError(
                f"chunk_overlap_tokens cannot be negative, got {chunk_overlap_tokens}"
            )
        if chunk_overlap_tokens >= chunk_size_tokens:
            raise ValueError(
                f"chunk_overlap_tokens ({chunk_overlap_tokens}) must be strictly less "
                f"than chunk_size_tokens ({chunk_size_tokens})"
            )

        self.chunk_size_tokens = chunk_size_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens
        self.encoding_name = encoding_name
        self.tokenizer = tiktoken.get_encoding(encoding_name)

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string.

        Args:
            text: Text to evaluate.

        Returns:
            int: Token count.
        """
        if not text:
            return 0
        return len(self.tokenizer.encode(text, disallowed_special=()))

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs.

        Allows raw text to contain special tokens (e.g. <|endofprompt|>) common in ML papers.

        Args:
            text: Text to encode.

        Returns:
            List[int]: List of token IDs.
        """
        return self.tokenizer.encode(text, disallowed_special=())

    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs back to a string.

        Args:
            tokens: List of token IDs.

        Returns:
            str: Decoded string.
        """
        return self.tokenizer.decode(tokens)

    def chunk(self, pages: List[Page]) -> List[Chunk]:
        """Split a collection of pages into chunks.

        Maintains page boundaries such that no single chunk spans across multiple pages.

        Args:
            pages: List of extracted document pages.

        Returns:
            List[Chunk]: Processed chunks ready for indexing.
        """
        all_chunks: List[Chunk] = []
        for page in pages:
            page_chunks = self.chunk_page(page)
            all_chunks.extend(page_chunks)
        return all_chunks

    @abstractmethod
    def chunk_page(self, page: Page) -> List[Chunk]:
        """Split a single page into retrieval chunks.

        Args:
            page: Document page to chunk.

        Returns:
            List[Chunk]: Chunks generated for this page.
        """
        pass


class FixedSizeChunker(Chunker):
    """Chunks text into fixed-size token windows with configurable overlap."""

    @classmethod
    def from_config(cls, config: IngestionConfig) -> "FixedSizeChunker":
        """Instantiate a FixedSizeChunker using an IngestionConfig instance.

        Args:
            config: Validated ingestion configuration.

        Returns:
            FixedSizeChunker: Configured chunker instance.
        """
        return cls(
            chunk_size_tokens=config.chunk_size_tokens,
            chunk_overlap_tokens=config.chunk_overlap_tokens,
            encoding_name=config.tokenizer_encoding,
        )

    def chunk_page(self, page: Page) -> List[Chunk]:
        """Split page text using fixed token window size and overlap.

        Args:
            page: Single document page.

        Returns:
            List[Chunk]: Fixed-size chunks with token counts <= chunk_size_tokens.
        """
        clean_text = page.text.strip()
        if not clean_text:
            return []

        tokens = self.encode(clean_text)
        if not tokens:
            return []

        # If page fits in a single chunk
        if len(tokens) <= self.chunk_size_tokens:
            return [
                Chunk(
                    doc_id=page.doc_id,
                    page_number=page.page_number,
                    chunk_id=f"{page.doc_id}_p{page.page_number}_c0",
                    text=clean_text,
                    token_count=len(tokens),
                )
            ]

        step = self.chunk_size_tokens - self.chunk_overlap_tokens
        chunks: List[Chunk] = []
        start = 0
        chunk_idx = 0

        while start < len(tokens):
            end = min(start + self.chunk_size_tokens, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.decode(chunk_tokens).strip()

            # Ensure stripped string does not exceed token limit due to BPE space boundaries
            while chunk_text and self.count_tokens(chunk_text) > self.chunk_size_tokens:
                chunk_tokens = chunk_tokens[:-1]
                chunk_text = self.decode(chunk_tokens).strip()

            if chunk_text:
                actual_tokens = self.count_tokens(chunk_text)
                chunks.append(
                    Chunk(
                        doc_id=page.doc_id,
                        page_number=page.page_number,
                        chunk_id=f"{page.doc_id}_p{page.page_number}_c{chunk_idx}",
                        text=chunk_text,
                        token_count=actual_tokens,
                    )
                )
                chunk_idx += 1

            if end == len(tokens):
                break

            start += step

        return chunks


class RecursiveChunker(Chunker):
    """Splits text hierarchically across paragraph, sentence, and word boundaries.

    Attempts to keep paragraphs together first. If paragraphs exceed the token limit,
    splits on sentence boundaries, then words, and finally token slices. Merges
    adjacent segments into chunks up to chunk_size_tokens while respecting chunk_overlap_tokens.
    """

    @classmethod
    def from_config(cls, config: IngestionConfig) -> "RecursiveChunker":
        """Instantiate a RecursiveChunker using an IngestionConfig instance.

        Args:
            config: Validated ingestion configuration.

        Returns:
            RecursiveChunker: Configured chunker instance.
        """
        return cls(
            chunk_size_tokens=config.chunk_size_tokens,
            chunk_overlap_tokens=config.chunk_overlap_tokens,
            encoding_name=config.tokenizer_encoding,
        )

    def _split_into_atomic_pieces(self, text: str) -> List[Piece]:
        """Recursively split text into atomic pieces where each piece <= chunk_size_tokens.

        Args:
            text: Raw text to split.

        Returns:
            List[Piece]: Pieces with appropriate preceding separators.
        """
        text = text.strip()
        if not text:
            return []

        if self.count_tokens(text) <= self.chunk_size_tokens:
            return [Piece(text, "")]

        # Level 1: Paragraphs (\n\n)
        if "\n\n" in text:
            paras = [p.strip() for p in text.split("\n\n") if p.strip()]
            pieces: List[Piece] = []
            for i, p in enumerate(paras):
                sub_pieces = self._split_into_atomic_pieces(p)
                if sub_pieces:
                    if i > 0:
                        sub_pieces[0].separator = "\n\n"
                    pieces.extend(sub_pieces)
            return pieces

        # Level 2: Line breaks (\n)
        if "\n" in text:
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            pieces: List[Piece] = []
            for i, l in enumerate(lines):
                sub_pieces = self._split_into_atomic_pieces(l)
                if sub_pieces:
                    if i > 0:
                        sub_pieces[0].separator = "\n"
                    pieces.extend(sub_pieces)
            return pieces

        # Level 3: Sentence boundaries (. ! ?)
        sentences = [
            s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()
        ]
        if len(sentences) > 1:
            pieces: List[Piece] = []
            for i, s in enumerate(sentences):
                sub_pieces = self._split_into_atomic_pieces(s)
                if sub_pieces:
                    if i > 0:
                        sub_pieces[0].separator = " "
                    pieces.extend(sub_pieces)
            return pieces

        # Level 4: Word boundaries
        words = [w.strip() for w in text.split(" ") if w.strip()]
        if len(words) > 1:
            pieces: List[Piece] = []
            for i, w in enumerate(words):
                sub_pieces = self._split_into_atomic_pieces(w)
                if sub_pieces:
                    if i > 0:
                        sub_pieces[0].separator = " "
                    pieces.extend(sub_pieces)
            return pieces

        # Level 5: Hard token split (single uninterrupted string larger than chunk_size)
        tokens = self.encode(text)
        pieces: List[Piece] = []
        for i in range(0, len(tokens), self.chunk_size_tokens):
            sub_tokens = tokens[i : i + self.chunk_size_tokens]
            pieces.append(Piece(self.decode(sub_tokens), ""))
        return pieces

    def _join_pieces(self, pieces: List[Piece]) -> str:
        """Combine a list of pieces into a cohesive text string."""
        if not pieces:
            return ""
        result: List[str] = []
        for i, piece in enumerate(pieces):
            if i == 0:
                result.append(piece.text)
            else:
                sep = piece.separator if piece.separator else " "
                result.append(sep + piece.text)
        return "".join(result)

    def _pack_pieces(self, pieces: List[Piece]) -> List[str]:
        """Pack atomic pieces into chunks respecting max tokens and overlap.

        Args:
            pieces: List of atomic pieces where each piece <= chunk_size_tokens.

        Returns:
            List[str]: Combined chunk texts.
        """
        chunks: List[str] = []
        current_pieces: List[Piece] = []
        i = 0

        while i < len(pieces):
            piece = pieces[i]
            candidate = current_pieces + [piece]
            candidate_text = self._join_pieces(candidate)

            if self.count_tokens(candidate_text) <= self.chunk_size_tokens:
                current_pieces.append(piece)
                i += 1
            else:
                if current_pieces:
                    chunk_text = self._join_pieces(current_pieces).strip()
                    if chunk_text:
                        chunks.append(chunk_text)

                    # Compute overlap pieces from current_pieces
                    overlap_pieces = list(current_pieces)
                    while (
                        overlap_pieces
                        and self.count_tokens(self._join_pieces(overlap_pieces))
                        > self.chunk_overlap_tokens
                    ):
                        overlap_pieces.pop(0)

                    # Guard against non-progression: drop at least one piece
                    if len(overlap_pieces) == len(current_pieces):
                        overlap_pieces.pop(0)

                    current_pieces = overlap_pieces
                else:
                    # Single piece fits exactly
                    chunks.append(piece.text.strip())
                    i += 1

        if current_pieces:
            chunk_text = self._join_pieces(current_pieces).strip()
            if chunk_text:
                chunks.append(chunk_text)

        return chunks

    def chunk_page(self, page: Page) -> List[Chunk]:
        """Split page hierarchically respecting linguistic boundaries.

        Args:
            page: Single document page.

        Returns:
            List[Chunk]: Chunks respecting paragraph and sentence boundaries.
        """
        clean_text = page.text.strip()
        if not clean_text:
            return []

        # If entire page fits in one chunk
        token_count = self.count_tokens(clean_text)
        if token_count <= self.chunk_size_tokens:
            return [
                Chunk(
                    doc_id=page.doc_id,
                    page_number=page.page_number,
                    chunk_id=f"{page.doc_id}_p{page.page_number}_c0",
                    text=clean_text,
                    token_count=token_count,
                )
            ]

        pieces = self._split_into_atomic_pieces(clean_text)
        chunk_texts = self._pack_pieces(pieces)

        chunks: List[Chunk] = []
        for idx, text in enumerate(chunk_texts):
            c_tokens = self.count_tokens(text)
            chunks.append(
                Chunk(
                    doc_id=page.doc_id,
                    page_number=page.page_number,
                    chunk_id=f"{page.doc_id}_p{page.page_number}_c{idx}",
                    text=text,
                    token_count=c_tokens,
                )
            )

        return chunks


def create_chunker(config: Optional[IngestionConfig] = None) -> Chunker:
    """Create a configured chunker instance based on ingestion configuration.

    Args:
        config: IngestionConfig instance. Defaults to default IngestionConfig().

    Returns:
        Chunker: Either FixedSizeChunker or RecursiveChunker.
    """
    cfg = config or IngestionConfig()
    if cfg.chunker == "recursive":
        return RecursiveChunker.from_config(cfg)
    return FixedSizeChunker.from_config(cfg)
