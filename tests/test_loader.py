"""Tests for the PDF document loader and text cleaning module."""

from pathlib import Path
import pymupdf
import pytest
from raglab.ingestion.loader import (
    Page,
    clean_text,
    detect_repeated_headers_and_footers,
    load_directory,
    load_pdf,
)


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    """Generate a clean, 3-page synthetic PDF with PyMuPDF for unit tests."""
    pdf_path = tmp_path / "sample_ml_paper.pdf"
    doc = pymupdf.open()
    doc.set_metadata({"title": "Attention Is All You Need"})

    # Page 1
    p1 = doc.new_page()
    p1.insert_text(
        (50, 50),
        "arXiv:1706.03762v7 [cs.CL] 2 Aug 2023\n"
        "Abstract: The dominant sequence transduction models are based on complex RNNs.\n\n"
        "We propose the Transformer, a model architec-\n"
        "ture eschewing recurrence.\n\n"
        "Page 1 of 3",
    )

    # Page 2
    p2 = doc.new_page()
    p2.insert_text(
        (50, 50),
        "arXiv:1706.03762v7 [cs.CL] 2 Aug 2023\n"
        "1 Introduction\n\n"
        "Recurrent neural networks have been firmly established as state-of-the-art.\n"
        "The fundamental constraint of sequential computation remains.\n\n"
        "Page 2 of 3",
    )

    # Page 3
    p3 = doc.new_page()
    p3.insert_text(
        (50, 50),
        "arXiv:1706.03762v7 [cs.CL] 2 Aug 2023\n"
        "2 Model Architecture\n\n"
        "The Transformer follows an encoder-decoder structure using stacked self-attention.\n\n"
        "Key components include:\n"
        "* Multi-head attention layers.\n"
        "* Point-wise feed-forward networks.\n\n"
        "Page 3 of 3",
    )

    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


def test_page_model_instantiation() -> None:
    """Verify that Page model instantiates with expected attributes."""
    page = Page(
        doc_id="test_001",
        title="Test Title",
        page_number=1,
        text="Extracted text content.",
    )
    assert page.doc_id == "test_001"
    assert page.title == "Test Title"
    assert page.page_number == 1
    assert page.text == "Extracted text content."


def test_load_pdf_extraction_and_page_numbering(sample_pdf: Path) -> None:
    """Verify page count, 1-based numbering, and text extraction."""
    pages = load_pdf(sample_pdf)

    assert len(pages) == 3
    for idx, page in enumerate(pages):
        assert page.page_number == idx + 1
        assert page.doc_id == "sample_ml_paper"
        assert page.title == "Attention Is All You Need"
        assert len(page.text) > 0


def test_title_metadata_and_fallback(tmp_path: Path) -> None:
    """Verify title uses PDF metadata if available, and falls back to filename stem."""
    # 1. PDF with metadata title
    path_with_title = tmp_path / "paper_alpha.pdf"
    doc1 = pymupdf.open()
    doc1.set_metadata({"title": "Transformers in Vision"})
    p = doc1.new_page()
    p.insert_text((50, 50), "Content here.")
    doc1.save(str(path_with_title))
    doc1.close()

    pages1 = load_pdf(path_with_title)
    assert pages1[0].title == "Transformers in Vision"
    assert pages1[0].doc_id == "paper_alpha"

    # 2. PDF without metadata title (empty title)
    path_without_title = tmp_path / "arxiv_2301_00000.pdf"
    doc2 = pymupdf.open()
    doc2.set_metadata({"title": ""})
    p = doc2.new_page()
    p.insert_text((50, 50), "Content here.")
    doc2.save(str(path_without_title))
    doc2.close()

    pages2 = load_pdf(path_without_title)
    assert pages2[0].title == "arxiv_2301_00000"
    assert pages2[0].doc_id == "arxiv_2301_00000"


def test_text_cleaning_hyphenation_and_paragraphs(sample_pdf: Path) -> None:
    """Verify hyphenated line-breaks are resolved and paragraphs preserved."""
    pages = load_pdf(sample_pdf)

    # In Page 1: "architec-\nture" should become "architecture"
    assert "architecture" in pages[0].text
    assert "architec-" not in pages[0].text

    # In Page 3: bullet points should be preserved on separate lines
    assert "* Multi-head attention layers." in pages[2].text
    assert "* Point-wise feed-forward networks." in pages[2].text


def test_repeated_headers_and_footers_removal(sample_pdf: Path) -> None:
    """Verify multi-page recurring headers and footers are detected and stripped."""
    pages = load_pdf(sample_pdf)

    for page in pages:
        # Repeated header should be stripped
        assert "arXiv:1706.03762v7" not in page.text
        # Running footer page number should be stripped
        assert f"Page {page.page_number} of 3" not in page.text

    # Unique body text should be retained
    assert "Abstract: The dominant sequence transduction models" in pages[0].text
    assert "Recurrent neural networks have been firmly established" in pages[1].text
    assert "The Transformer follows an encoder-decoder structure" in pages[2].text


def test_clean_text_direct() -> None:
    """Test clean_text function directly on various formatting challenges."""
    raw = (
        "This is a sentence with hyphen-\nated words and extra   horizontal   spaces.\n\n"
        "Here is the second paragraph.\nIt continues on a new line.\n\n"
        "- Bullet 1\n"
        "- Bullet 2\n"
    )
    cleaned = clean_text(raw)
    assert "hyphenated words" in cleaned
    assert "extra horizontal spaces." in cleaned
    assert "Here is the second paragraph. It continues on a new line." in cleaned
    assert "- Bullet 1\n- Bullet 2" in cleaned


def test_corrupted_pdf_handling(tmp_path: Path) -> None:
    """Verify corrupted PDF files are handled gracefully without raising exceptions."""
    corrupt_file = tmp_path / "corrupt.pdf"
    corrupt_file.write_bytes(b"%PDF-1.5 \n Invalid garbage content that cannot be parsed")

    pages = load_pdf(corrupt_file)
    assert pages == []


def test_empty_pdf_handling(tmp_path: Path) -> None:
    """Verify 0-byte PDF files return empty list without crashing."""
    empty_file = tmp_path / "empty.pdf"
    empty_file.write_bytes(b"")

    pages = load_pdf(empty_file)
    assert pages == []


def test_encrypted_pdf_handling(tmp_path: Path) -> None:
    """Verify encrypted password-protected PDFs return empty list with warning."""
    encrypted_file = tmp_path / "encrypted.pdf"
    doc = pymupdf.open()
    p = doc.new_page()
    p.insert_text((50, 50), "Top secret content.")
    doc.save(
        str(encrypted_file),
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        user_pw="super_secret_password",
        owner_pw="super_secret_password",
    )
    doc.close()

    pages = load_pdf(encrypted_file)
    assert pages == []


def test_non_existent_file() -> None:
    """Verify non-existent file returns empty list."""
    pages = load_pdf("non_existent_file_xyz.pdf")
    assert pages == []


def test_load_directory_mixed_files(tmp_path: Path) -> None:
    """Verify load_directory loads valid PDFs and skips corrupted ones without crashing."""
    valid_dir = tmp_path / "papers_batch"
    valid_dir.mkdir()

    # Create 2 valid PDFs
    for name in ["paper_1.pdf", "paper_2.pdf"]:
        doc = pymupdf.open()
        p = doc.new_page()
        p.insert_text((50, 50), f"Content of {name}")
        doc.save(str(valid_dir / name))
        doc.close()

    # Create 1 corrupt PDF
    (valid_dir / "paper_bad.pdf").write_bytes(b"Corrupt data")

    # Create 1 non-pdf file
    (valid_dir / "notes.txt").write_text("Not a pdf file")

    pages = load_directory(valid_dir)
    # 2 valid PDFs with 1 page each = 2 pages
    assert len(pages) == 2
    doc_ids = {p.doc_id for p in pages}
    assert doc_ids == {"paper_1", "paper_2"}


def test_load_directory_invalid_path() -> None:
    """Verify non-existent directory returns empty list."""
    pages = load_directory("non_existent_directory_abc")
    assert pages == []
