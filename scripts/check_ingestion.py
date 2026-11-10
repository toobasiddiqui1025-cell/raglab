#!/usr/bin/env python3
"""Ingestion verification script for Inquira / RAGLab.

Discovers research paper PDFs in data/papers/, parses them into structured pages
using the PyMuPDF loader, splits pages into chunks using RecursiveChunker,
and reports comprehensive summary metrics on pages, chunks, and token distributions.

Usage:
    python scripts/check_ingestion.py
    python scripts/check_ingestion.py --papers-dir data/papers --config configs/baseline.yaml
    python scripts/check_ingestion.py --verbose
"""

import argparse
import logging
from pathlib import Path
import sys
from typing import List, Tuple

from raglab.config import AppConfig, load_config
from raglab.ingestion.chunker import Chunk, RecursiveChunker
from raglab.ingestion.loader import Page, load_pdf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("check_ingestion")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Verify document ingestion and chunking over PDF corpus."
    )
    parser.add_argument(
        "--papers-dir",
        type=Path,
        default=Path("data/papers"),
        help="Directory containing PDF files (default: data/papers).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/baseline.yaml"),
        help="Path to pipeline configuration YAML (default: configs/baseline.yaml).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed per-document processing statistics.",
    )
    return parser.parse_args()


def run_ingestion_check(
    papers_dir: Path,
    config_path: Path,
    verbose: bool = False,
) -> int:
    """Run ingestion and chunking verification across discovered PDFs.

    Args:
        papers_dir: Path to directory containing PDF files.
        config_path: Path to configuration YAML file.
        verbose: Whether to log per-paper progress.

    Returns:
        int: Exit status code (0 for success, non-zero for failure).
    """
    if not papers_dir.is_dir():
        logger.error("Papers directory does not exist: %s", papers_dir)
        return 1

    # Load configuration
    try:
        app_config: AppConfig = load_config(config_path)
        logger.info(
            "Loaded configuration from '%s' (chunk_size=%d, overlap=%d, encoding='%s')",
            config_path,
            app_config.ingestion.chunk_size_tokens,
            app_config.ingestion.chunk_overlap_tokens,
            app_config.ingestion.tokenizer_encoding,
        )
    except Exception as e:
        logger.error("Failed to load configuration from '%s': %s", config_path, e)
        return 1

    # Instantiate RecursiveChunker from project configuration
    chunker = RecursiveChunker.from_config(app_config.ingestion)

    # Discover PDFs
    pdf_files = sorted(papers_dir.glob("*.pdf"))
    total_discovered = len(pdf_files)
    logger.info("Discovered %d PDF files in '%s'.", total_discovered, papers_dir)

    if total_discovered == 0:
        logger.warning("No PDF files found to process.")
        return 0

    successful_pdfs: List[Path] = []
    failed_pdfs: List[Tuple[Path, str]] = []
    total_pages = 0
    all_chunks: List[Chunk] = []

    print("\nProcessing papers...")
    for idx, pdf_path in enumerate(pdf_files, 1):
        try:
            pages: List[Page] = load_pdf(pdf_path)
            if not pages:
                failed_pdfs.append((pdf_path, "No pages extracted (unreadable or empty)"))
                if verbose:
                    print(f"[{idx}/{total_discovered}] ❌ {pdf_path.name}: 0 pages extracted")
                continue

            chunks: List[Chunk] = chunker.chunk(pages)
            successful_pdfs.append(pdf_path)
            total_pages += len(pages)
            all_chunks.extend(chunks)

            if verbose:
                avg_tok = (
                    sum(c.token_count for c in chunks) / len(chunks) if chunks else 0.0
                )
                print(
                    f"[{idx}/{total_discovered}] ✅ {pdf_path.name}: "
                    f"{len(pages)} pages -> {len(chunks)} chunks (avg {avg_tok:.1f} tokens)"
                )

        except Exception as e:
            failed_pdfs.append((pdf_path, str(e)))
            logger.error("Error processing '%s': %s", pdf_path.name, e)

    # Compute summary statistics
    total_chunks = len(all_chunks)
    chunk_tokens = [c.token_count for c in all_chunks]
    avg_tokens = sum(chunk_tokens) / total_chunks if total_chunks > 0 else 0.0
    min_tokens = min(chunk_tokens) if total_chunks > 0 else 0
    max_tokens = max(chunk_tokens) if total_chunks > 0 else 0

    # Print summary report
    print("\n" + "=" * 60)
    print("           INQUIRA INGESTION VERIFICATION REPORT           ")
    print("=" * 60)
    print(f"  • Total PDFs discovered:         {total_discovered}")
    print(f"  • Successfully processed PDFs:   {len(successful_pdfs)}")
    print(f"  • Failed PDFs:                   {len(failed_pdfs)}")
    print(f"  • Total pages extracted:         {total_pages}")
    print(f"  • Total chunks generated:        {total_chunks}")
    print(f"  • Average tokens per chunk:      {avg_tokens:.1f}")
    print(f"  • Minimum chunk token count:     {min_tokens}")
    print(f"  • Maximum chunk token count:     {max_tokens}")
    print("=" * 60)

    if failed_pdfs:
        print("\nFailed PDFs Summary:")
        for failed_path, reason in failed_pdfs:
            print(f"  - {failed_path.name}: {reason}")
        print("=" * 60)

    return 0


def main() -> None:
    """CLI entrypoint."""
    args = parse_arguments()
    sys.exit(
        run_ingestion_check(
            papers_dir=args.papers_dir,
            config_path=args.config,
            verbose=args.verbose,
        )
    )


if __name__ == "__main__":
    main()
