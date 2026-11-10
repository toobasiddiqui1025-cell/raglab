"""PDF document loader and text extraction module.

Provides functionality to parse ML research papers from PDF format into structured,
page-aware Page objects using PyMuPDF (fitz). Cleans text by stripping repeated
headers/footers, normalizing line-breaks, and collapsing whitespace.
"""

from pathlib import Path
from typing import List
from pydantic import BaseModel


class Page(BaseModel):
    """Represents a single extracted page from a document.

    Attributes:
        doc_id: Unique identifier for the source paper (e.g., arXiv ID or filename).
        title: Title of the document.
        page_number: 1-indexed page number within the original document.
        text: Cleaned text content of the page.
    """


def clean_text(raw_text: str) -> str:
    """Clean extracted page text by fixing hyphenations and normalizing whitespace.

    Args:
        raw_text: Raw text string extracted directly from PDF page.

    Returns:
        str: Normalized and cleaned text content.
    """
    raise NotImplementedError


def load_pdf(file_path: Path) -> List[Page]:
    """Extract page objects from a single PDF document.

    Args:
        file_path: Absolute or relative path to the PDF file.

    Returns:
        List[Page]: Extracted and cleaned pages with metadata.
    """
    raise NotImplementedError


def load_directory(directory_path: Path) -> List[Page]:
    """Batch load and parse all PDF documents in a given directory.

    Args:
        directory_path: Path to directory containing PDF files.

    Returns:
        List[Page]: Aggregated list of all extracted pages across documents.
    """
    raise NotImplementedError
