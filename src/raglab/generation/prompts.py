"""Prompt engineering and template definitions for grounded generation.

Defines strict system instructions and formatting templates ensuring answers
derive solely from retrieved context chunks, include page-level citations in
[doc_id, p.N] format, and refuse unanswerable queries cleanly.
"""

from typing import List
from raglab.indexing.vector_store import SearchResult

SYSTEM_PROMPT: str = (
    "You are a precise research assistant answering questions using only the provided excerpts "
    "from machine learning research papers. You must strictly follow these rules:\n"
    "1. Answer using ONLY the information provided in the context chunks.\n"
    "2. For every claim you make, cite the source in the format [doc_id, p.N] where N is the page number.\n"
    "3. If the context does not contain enough information to answer, reply exactly: "
    "'I could not find this in the provided documents.' Do not extrapolate or fabricate."
)


def format_context_chunks(chunks: List[SearchResult]) -> str:
    """Format retrieved search result chunks into structured context blocks for the prompt.

    Args:
        chunks: List of retrieved SearchResult objects.

    Returns:
        str: Formatted context string with document IDs and page numbers clearly marked.
    """
    raise NotImplementedError


def build_rag_prompt(question: str, chunks: List[SearchResult]) -> str:
    """Combine user question and formatted context chunks into a full RAG prompt.

    Args:
        question: User inquiry string.
        chunks: Retrieved context chunks.

    Returns:
        str: Formatted user prompt ready for LLM consumption.
    """
    raise NotImplementedError
