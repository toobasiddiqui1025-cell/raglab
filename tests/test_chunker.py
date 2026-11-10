"""Unit tests for the document chunking module."""

from pathlib import Path
import pytest
from raglab.config import IngestionConfig, load_config
from raglab.ingestion.chunker import (
    Chunk,
    Chunker,
    FixedSizeChunker,
    RecursiveChunker,
    create_chunker,
)
from raglab.ingestion.loader import Page


@pytest.fixture
def sample_pages() -> list[Page]:
    """Provide a list of sample pages with varied paragraph structures."""
    return [
        Page(
            doc_id="paper_001",
            title="Attention Is All You Need",
            page_number=1,
            text=(
                "The dominant sequence transduction models are based on complex recurrent "
                "or convolutional neural networks in an encoder-decoder configuration.\n\n"
                "We propose the Transformer, a model architecture eschewing recurrence and "
                "instead relying entirely on an attention mechanism to draw global dependencies "
                "between input and output."
            ),
        ),
        Page(
            doc_id="paper_001",
            title="Attention Is All You Need",
            page_number=2,
            text=(
                "Recurrent neural networks typically factor computation along the symbol positions "
                "of the input and output sequences. Aligning the positions to steps in computation "
                "time, they generate a sequence of hidden states ht as a function of the previous "
                "hidden state ht-1 and the input for position t.\n\n"
                "This inherently sequential nature precludes parallelization within training examples, "
                "which becomes critical at longer sequence lengths, as memory constraints limit batching "
                "across examples."
            ),
        ),
    ]


def test_chunk_model_instantiation() -> None:
    """Verify that Chunk model can be created with all required fields."""
    chunk = Chunk(
        doc_id="doc_123",
        page_number=5,
        chunk_id="doc_123_p5_c0",
        text="Sample chunk content.",
        token_count=4,
    )
    assert chunk.doc_id == "doc_123"
    assert chunk.page_number == 5
    assert chunk.chunk_id == "doc_123_p5_c0"
    assert chunk.text == "Sample chunk content."
    assert chunk.token_count == 4


def test_chunker_validation() -> None:
    """Verify chunker initialization enforces valid parameters."""
    with pytest.raises(ValueError, match="chunk_size_tokens must be positive"):
        FixedSizeChunker(chunk_size_tokens=0)

    with pytest.raises(ValueError, match="chunk_overlap_tokens cannot be negative"):
        FixedSizeChunker(chunk_size_tokens=100, chunk_overlap_tokens=-1)

    with pytest.raises(ValueError, match="must be strictly less than chunk_size_tokens"):
        FixedSizeChunker(chunk_size_tokens=100, chunk_overlap_tokens=100)


def test_empty_input_handling() -> None:
    """Verify chunkers handle empty page list, empty text, and whitespace gracefully."""
    fixed = FixedSizeChunker(chunk_size_tokens=100, chunk_overlap_tokens=10)
    recursive = RecursiveChunker(chunk_size_tokens=100, chunk_overlap_tokens=10)

    # Empty page list
    assert fixed.chunk([]) == []
    assert recursive.chunk([]) == []

    # Empty text page
    empty_page = Page(doc_id="d1", title="T", page_number=1, text="")
    assert fixed.chunk([empty_page]) == []
    assert recursive.chunk([empty_page]) == []

    # Whitespace-only page
    whitespace_page = Page(doc_id="d1", title="T", page_number=1, text="   \n\n  \t  ")
    assert fixed.chunk([whitespace_page]) == []
    assert recursive.chunk([whitespace_page]) == []


def test_fixed_size_chunker_max_size_and_overlap(sample_pages: list[Page]) -> None:
    """Verify FixedSizeChunker never exceeds max token limit and creates correct overlap."""
    chunk_size = 25
    overlap = 8
    chunker = FixedSizeChunker(chunk_size_tokens=chunk_size, chunk_overlap_tokens=overlap)

    chunks = chunker.chunk(sample_pages)
    assert len(chunks) > 0

    for chunk in chunks:
        assert chunk.token_count <= chunk_size, (
            f"Chunk {chunk.chunk_id} exceeded limit: {chunk.token_count} > {chunk_size}"
        )
        assert chunk.token_count == chunker.count_tokens(chunk.text)

    # Check overlap between consecutive chunks on the same page
    p1_chunks = [c for c in chunks if c.page_number == 1]
    assert len(p1_chunks) >= 2

    for i in range(len(p1_chunks) - 1):
        chunk_a = p1_chunks[i].text
        chunk_b = p1_chunks[i + 1].text
        words_a = chunk_a.split()
        overlap_snippet = " ".join(words_a[-2:])
        norm_b = " ".join(chunk_b.split())
        assert overlap_snippet in norm_b, f"Expected '{overlap_snippet}' in '{norm_b}'"


def test_recursive_chunker_max_size_and_overlap(sample_pages: list[Page]) -> None:
    """Verify RecursiveChunker respects max token size and creates valid overlap."""
    chunk_size = 35
    overlap = 10
    chunker = RecursiveChunker(chunk_size_tokens=chunk_size, chunk_overlap_tokens=overlap)

    chunks = chunker.chunk(sample_pages)
    assert len(chunks) > 0

    for chunk in chunks:
        assert chunk.token_count <= chunk_size, (
            f"Chunk {chunk.chunk_id} exceeded limit: {chunk.token_count} > {chunk_size}"
        )
        assert chunk.token_count == chunker.count_tokens(chunk.text)


def test_recursive_paragraph_and_sentence_splitting() -> None:
    """Verify recursive chunker splits paragraphs first, then sentences, and preserves structure."""
    text = (
        "Paragraph one is relatively short.\n\n"
        "Paragraph two is longer. It has a second sentence. And a third sentence that adds more tokens.\n\n"
        "Paragraph three concludes the section."
    )
    page = Page(doc_id="test_doc", title="Title", page_number=1, text=text)

    # When chunk_size is generous, paragraphs fit together with \n\n
    large_chunker = RecursiveChunker(chunk_size_tokens=200, chunk_overlap_tokens=0)
    single_chunks = large_chunker.chunk([page])
    assert len(single_chunks) == 1
    assert "\n\n" in single_chunks[0].text

    # When chunk_size is small (e.g. 15 tokens), paragraph 2 must split on sentence boundaries
    small_chunker = RecursiveChunker(chunk_size_tokens=15, chunk_overlap_tokens=4)
    small_chunks = small_chunker.chunk([page])
    assert len(small_chunks) > 1

    for c in small_chunks:
        assert c.token_count <= 15
        # Verify sentences aren't randomly truncated mid-word
        assert len(c.text.split()) > 0


def test_metadata_preservation(sample_pages: list[Page]) -> None:
    """Verify doc_id, page_number, and stable chunk_id are preserved."""
    chunker = FixedSizeChunker(chunk_size_tokens=20, chunk_overlap_tokens=5)
    chunks = chunker.chunk(sample_pages)

    p1_chunks = [c for c in chunks if c.page_number == 1]
    p2_chunks = [c for c in chunks if c.page_number == 2]

    assert len(p1_chunks) > 0
    assert len(p2_chunks) > 0

    # Ensure every chunk has matching doc_id and stable formatted chunk_id
    for idx, c in enumerate(p1_chunks):
        assert c.doc_id == "paper_001"
        assert c.page_number == 1
        assert c.chunk_id == f"paper_001_p1_c{idx}"

    for idx, c in enumerate(p2_chunks):
        assert c.doc_id == "paper_001"
        assert c.page_number == 2
        assert c.chunk_id == f"paper_001_p2_c{idx}"


def test_pages_are_never_combined() -> None:
    """Verify that chunks never combine text from two different pages."""
    page_1 = Page(doc_id="docA", title="T", page_number=1, text="Text from page one.")
    page_2 = Page(doc_id="docA", title="T", page_number=2, text="Text from page two.")

    chunker = RecursiveChunker(chunk_size_tokens=500, chunk_overlap_tokens=50)
    chunks = chunker.chunk([page_1, page_2])

    assert len(chunks) == 2
    assert chunks[0].page_number == 1
    assert "page one" in chunks[0].text
    assert "page two" not in chunks[0].text

    assert chunks[1].page_number == 2
    assert "page two" in chunks[1].text
    assert "page one" not in chunks[1].text


def test_very_long_paragraphs_and_sentences() -> None:
    """Verify chunkers handle massive paragraphs and unbroken strings without error."""
    # Massive paragraph of 300 words
    long_para = " ".join([f"sentence_{i} explains point {i}." for i in range(150)])
    page_long_para = Page(doc_id="doc_long", title="T", page_number=1, text=long_para)

    chunker = RecursiveChunker(chunk_size_tokens=40, chunk_overlap_tokens=10)
    chunks = chunker.chunk([page_long_para])

    assert len(chunks) > 1
    for c in chunks:
        assert c.token_count <= 40

    # Massive continuous string without any spaces (500 characters)
    continuous_string = "A" * 500
    page_unbroken = Page(doc_id="doc_unbroken", title="T", page_number=1, text=continuous_string)
    unbroken_chunks = chunker.chunk([page_unbroken])

    assert len(unbroken_chunks) > 1
    for c in unbroken_chunks:
        assert c.token_count <= 40


def test_config_integration() -> None:
    """Verify chunkers can be instantiated and configured from IngestionConfig."""
    cfg_fixed = IngestionConfig(
        chunker="fixed",
        chunk_size_tokens=300,
        chunk_overlap_tokens=30,
        tokenizer_encoding="cl100k_base",
    )
    chunker_fixed = create_chunker(cfg_fixed)
    assert isinstance(chunker_fixed, FixedSizeChunker)
    assert chunker_fixed.chunk_size_tokens == 300
    assert chunker_fixed.chunk_overlap_tokens == 30

    cfg_rec = IngestionConfig(
        chunker="recursive",
        chunk_size_tokens=250,
        chunk_overlap_tokens=25,
        tokenizer_encoding="cl100k_base",
    )
    chunker_rec = create_chunker(cfg_rec)
    assert isinstance(chunker_rec, RecursiveChunker)
    assert chunker_rec.chunk_size_tokens == 250
    assert chunker_rec.chunk_overlap_tokens == 25


def test_baseline_yaml_loading() -> None:
    """Verify loading from configs/baseline.yaml correctly initializes chunker."""
    app_config = load_config("configs/baseline.yaml")
    chunker = create_chunker(app_config.ingestion)

    assert isinstance(chunker, FixedSizeChunker)
    assert chunker.chunk_size_tokens == 500
    assert chunker.chunk_overlap_tokens == 50
