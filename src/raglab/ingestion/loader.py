"""PDF document loader and text extraction module.

Provides functionality to parse ML research papers from PDF format into structured,
page-aware Page objects using PyMuPDF. Cleans text by stripping repeated
headers/footers, normalizing line-breaks, collapsing excessive whitespace,
and preserving paragraph structure.
"""

import logging
from pathlib import Path
import re
from typing import List, Optional, Set, Tuple, Union
import pymupdf
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Pattern matching standalone page numbers or running page indicators
PAGE_NUM_PATTERN = re.compile(
    r"^(?:(?:page\s+)?\d+(?:\s*(?:of|/|-)\s*\d+)?|-\s*\d+\s*-|[ivxlcdm]+)$",
    re.IGNORECASE,
)

# Pattern matching bullet points, numbered lists, or lettered lists
LIST_ITEM_PATTERN = re.compile(r"^([-*•]|\d+\.|\([a-zA-Z0-9]+\))\s+")


class Page(BaseModel):
    """Represents a single extracted page from a document.

    Attributes:
        doc_id: Unique identifier for the source paper (e.g., arXiv ID or filename stem).
        title: Title of the document from metadata or fallback to filename.
        page_number: 1-indexed page number within the original document.
        text: Cleaned text content of the page.
    """

    doc_id: str
    title: str
    page_number: int
    text: str


def is_page_number_line(line: str, page_num: int, is_top: bool) -> bool:
    """Determine whether a line at the top or bottom of a page represents a page number.

    Args:
        line: The text line to evaluate.
        page_num: 1-based page number of the current page.
        is_top: True if the line appears near the top of the page, False if near bottom.

    Returns:
        bool: True if the line represents a page number indicator, False otherwise.
    """
    clean_line = line.strip()
    if not clean_line:
        return False

    # Check explicit patterns like "Page 1", "page 1 of 5", "- 2 -", or roman numerals
    if PAGE_NUM_PATTERN.match(clean_line):
        digits_match = re.match(r"^(\d+)$", clean_line)
        if digits_match and is_top:
            # At top of page, only treat as page number if it matches current page,
            # preventing false positives on section headings like "1" or "2".
            return int(digits_match.group(1)) == page_num
        return True

    return False


def detect_repeated_headers_and_footers(
    raw_pages: List[str],
) -> Tuple[Set[str], Set[str]]:
    """Detect recurring running headers and footers across pages in a multi-page document.

    Args:
        raw_pages: List of raw string text extracted for each page.

    Returns:
        Tuple[Set[str], Set[str]]: Sets of detected repeated header and footer strings.
    """
    total_pages = len(raw_pages)
    if total_pages < 2:
        return set(), set()

    header_counts: dict[str, int] = {}
    footer_counts: dict[str, int] = {}
    # Threshold: must appear on at least 2 pages and at least 35% of all pages
    threshold = max(2, int(total_pages * 0.35))

    for idx, raw in enumerate(raw_pages):
        page_num = idx + 1
        lines = [l.strip() for l in raw.split("\n") if l.strip()]
        if not lines:
            continue

        # Check top-most candidate line
        top_line = lines[0]
        if len(top_line) <= 150 and not is_page_number_line(top_line, page_num, is_top=True):
            header_counts[top_line] = header_counts.get(top_line, 0) + 1

        # Check bottom-most candidate line
        bottom_line = lines[-1]
        if len(bottom_line) <= 150 and not is_page_number_line(bottom_line, page_num, is_top=False):
            footer_counts[bottom_line] = footer_counts.get(bottom_line, 0) + 1

    repeated_headers = {h for h, count in header_counts.items() if count >= threshold}
    repeated_footers = {f for f, count in footer_counts.items() if count >= threshold}
    return repeated_headers, repeated_footers


def strip_headers_and_footers(
    lines: List[str],
    repeated_headers: Set[str],
    repeated_footers: Set[str],
    page_num: int,
) -> List[str]:
    """Strip detected running headers, footers, and page numbers from line list.

    Args:
        lines: Lines of the page text.
        repeated_headers: Set of detected recurring header strings.
        repeated_footers: Set of detected recurring footer strings.
        page_num: 1-indexed page number.

    Returns:
        List[str]: Filtered list of lines with headers and footers removed.
    """
    non_empty_indices = [i for i, line in enumerate(lines) if line.strip()]
    if not non_empty_indices:
        return lines

    indices_to_remove: Set[int] = set()

    # Inspect top candidates (up to first 2 non-empty lines)
    for idx in non_empty_indices[:2]:
        clean_line = lines[idx].strip()
        if clean_line in repeated_headers or is_page_number_line(clean_line, page_num, is_top=True):
            indices_to_remove.add(idx)
        else:
            # If the outermost line is not a header, stop peeling
            break

    # Inspect bottom candidates (up to last 2 non-empty lines)
    for idx in reversed(non_empty_indices[-2:]):
        if idx in indices_to_remove:
            continue
        clean_line = lines[idx].strip()
        if clean_line in repeated_footers or is_page_number_line(clean_line, page_num, is_top=False):
            indices_to_remove.add(idx)
        else:
            # If the bottommost line is not a footer, stop peeling
            break

    return [line for i, line in enumerate(lines) if i not in indices_to_remove]


def clean_paragraph(paragraph: str) -> str:
    """Normalize whitespace and join wrapped sentence lines inside a single paragraph.

    Args:
        paragraph: Raw paragraph text block.

    Returns:
        str: Cleaned paragraph string.
    """
    lines = [l.strip() for l in paragraph.split("\n") if l.strip()]
    if not lines:
        return ""

    cleaned_lines = [re.sub(r"[ \t]+", " ", line) for line in lines]
    result: List[str] = []

    for i, line in enumerate(cleaned_lines):
        if i == 0:
            result.append(line)
        else:
            # Preserve line breaks for list items/bullets
            if LIST_ITEM_PATTERN.match(line):
                result.append("\n" + line)
            else:
                result.append(" " + line)

    return "".join(result)


def clean_text(
    raw_text: str,
    repeated_headers: Optional[Set[str]] = None,
    repeated_footers: Optional[Set[str]] = None,
    page_num: int = 1,
) -> str:
    """Clean extracted page text by removing headers/footers, fixing hyphenations,
    normalizing whitespace, and preserving paragraph breaks.

    Args:
        raw_text: Raw text string extracted directly from PDF page.
        repeated_headers: Optional set of detected repeated header strings to strip.
        repeated_footers: Optional set of detected repeated footer strings to strip.
        page_num: 1-indexed page number for contextual page-number detection.

    Returns:
        str: Normalized and cleaned text content.
    """
    if not raw_text:
        return ""

    # Normalize carriage returns and non-breaking spaces
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")

    lines = text.split("\n")
    headers_set = repeated_headers or set()
    footers_set = repeated_footers or set()

    # Strip detected headers, footers, and page numbers
    filtered_lines = strip_headers_and_footers(lines, headers_set, footers_set, page_num)
    text = "\n".join(filtered_lines)

    # Fix hyphenated line breaks (e.g. "trans-\nformer" -> "transformer")
    text = re.sub(r"([a-zA-Z]+)-\s*\n\s*([a-zA-Z]+)", r"\1\2", text)

    # Segment into paragraphs by blank lines (\n\s*\n+)
    paragraphs = re.split(r"\n\s*\n+", text)
    cleaned_paragraphs = [clean_paragraph(p) for p in paragraphs]

    return "\n\n".join(p for p in cleaned_paragraphs if p).strip()


def load_pdf(
    file_path: Union[str, Path],
    doc_id: Optional[str] = None,
) -> List[Page]:
    """Extract page objects from a single PDF document.

    Args:
        file_path: Absolute or relative path to the PDF file.
        doc_id: Optional unique identifier for the document. Defaults to file stem.

    Returns:
        List[Page]: Extracted and cleaned pages with metadata, or empty list on error.
    """
    path = Path(file_path)
    if not path.is_file():
        logger.error("PDF file not found: '%s'", path)
        return []

    assigned_doc_id = doc_id or path.stem

    try:
        doc = pymupdf.open(path)
    except pymupdf.EmptyFileError:
        logger.warning("PDF file is empty (0 bytes): '%s'", path)
        return []
    except pymupdf.FileDataError as e:
        logger.error("Corrupted or invalid PDF format in '%s': %s", path, e)
        return []
    except Exception as e:
        logger.error("Failed to open PDF '%s': %s", path, e)
        return []

    try:
        # Check for encryption or password protection
        if doc.needs_pass or doc.is_encrypted:
            authenticated = False
            try:
                authenticated = doc.authenticate("")
            except Exception:
                pass

            if not authenticated:
                logger.warning(
                    "PDF '%s' is password-protected or encrypted; skipping.", path
                )
                return []

        if len(doc) == 0:
            logger.warning("PDF '%s' contains 0 pages; skipping.", path)
            return []

        # Determine document title from PDF metadata or fallback to filename stem
        raw_title = doc.metadata.get("title") if doc.metadata else None
        if (
            raw_title
            and raw_title.strip()
            and raw_title.strip().lower() not in {"untitled", "none"}
        ):
            title = raw_title.strip()
        else:
            title = path.stem

        # Extract raw text from each page
        raw_pages_text: List[str] = []
        for page_idx in range(len(doc)):
            try:
                page = doc[page_idx]
                raw_pages_text.append(page.get_text("text") or "")
            except Exception as e:
                logger.warning(
                    "Error extracting text from page %d of '%s': %s",
                    page_idx + 1,
                    path,
                    e,
                )
                raw_pages_text.append("")

        # Identify repeated headers and footers across the document
        repeated_headers, repeated_footers = detect_repeated_headers_and_footers(
            raw_pages_text
        )

        pages: List[Page] = []
        for idx, raw_text in enumerate(raw_pages_text):
            page_num = idx + 1
            cleaned = clean_text(
                raw_text,
                repeated_headers=repeated_headers,
                repeated_footers=repeated_footers,
                page_num=page_num,
            )
            pages.append(
                Page(
                    doc_id=assigned_doc_id,
                    title=title,
                    page_number=page_num,
                    text=cleaned,
                )
            )

        logger.info(
            "Successfully loaded '%s' (%s): %d pages extracted.",
            path.name,
            assigned_doc_id,
            len(pages),
        )
        return pages

    except Exception as e:
        logger.error("Unexpected error processing PDF '%s': %s", path, e)
        return []
    finally:
        try:
            doc.close()
        except Exception:
            pass


def load_directory(
    directory_path: Union[str, Path],
    recursive: bool = False,
) -> List[Page]:
    """Batch load and parse all PDF documents in a given directory.

    Args:
        directory_path: Path to directory containing PDF files.
        recursive: Whether to search subdirectories recursively.

    Returns:
        List[Page]: Aggregated list of all extracted pages across documents.
    """
    dir_path = Path(directory_path)
    if not dir_path.is_dir():
        logger.error("Directory not found or is not a directory: '%s'", dir_path)
        return []

    pattern = "**/*.pdf" if recursive else "*.pdf"
    pdf_files = sorted(dir_path.glob(pattern))

    if not pdf_files:
        logger.warning("No PDF files found in directory: '%s'", dir_path)
        return []

    all_pages: List[Page] = []
    for pdf_file in pdf_files:
        try:
            pages = load_pdf(pdf_file)
            all_pages.extend(pages)
        except Exception as e:
            logger.error("Unhandled exception loading '%s': %s", pdf_file, e)

    logger.info(
        "Batch ingestion complete for '%s': loaded %d pages across %d files.",
        dir_path,
        len(all_pages),
        len(pdf_files),
    )
    return all_pages
