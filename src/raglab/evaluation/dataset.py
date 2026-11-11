"""Gold evaluation dataset schema and loader for Inquira.

Defines two Pydantic models for the evaluation benchmark:

- GoldEvidence: A single piece of page-level evidence pointing to a specific
  document and one or more 1-indexed page numbers within that document.

- GoldQuestion: A complete benchmark question with its reference answer,
  semantic type, answerability flag, and ordered list of GoldEvidence entries.

Validation is split into two layers:

  1. Per-model (Pydantic field and model validators): enforce internal
     consistency that requires no external state — non-empty strings, positive
     page integers, cross-field rules (unanswerable ↔ empty evidence),
     duplicate page deduplication, and duplicate doc_id rejection.

  2. Dataset-level (load_gold_dataset): enforce corpus consistency — that every
     referenced doc_id exists in the PDF corpus and every referenced page
     number falls within the document's actual page range.

Page numbers follow the 1-indexed convention established by the PDF loader
(loader.py assigns page_number = idx + 1).
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, field_validator, model_validator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type alias for the corpus registry
# ---------------------------------------------------------------------------
# Maps doc_id (arXiv stem, e.g. "1706.03762") → total page count (int).
# Built from the actual PDFs at application startup or test fixture setup.
CorpusRegistry = Dict[str, int]


# ---------------------------------------------------------------------------
# GoldEvidence
# ---------------------------------------------------------------------------

class GoldEvidence(BaseModel):
    """A single piece of page-level evidence within a benchmark question.

    Points to one document and one or more specific pages within it.
    Pages are 1-indexed, matching the convention used by the PDF loader.

    Attributes:
        doc_id: Corpus document identifier (arXiv stem, e.g. '1706.03762').
        pages: Non-empty, sorted list of distinct 1-indexed page numbers
               containing the evidence for this question.
    """

    doc_id: str
    pages: List[int]

    @field_validator("doc_id")
    @classmethod
    def doc_id_not_empty(cls, v: str) -> str:
        """Reject empty or whitespace-only doc_id strings."""
        if not v or not v.strip():
            raise ValueError("doc_id must be a non-empty string.")
        return v.strip()

    @field_validator("pages")
    @classmethod
    def pages_valid(cls, v: List[int]) -> List[int]:
        """Validate, deduplicate, and sort page numbers.

        Rules:
        - At least one page must be specified.
        - All page numbers must be positive integers (>= 1), matching
          the 1-indexed convention of the PDF loader.
        - Duplicate pages are silently deduplicated and sorted ascending.
        """
        if not v:
            raise ValueError("pages must contain at least one page number.")

        for p in v:
            if p < 1:
                raise ValueError(
                    f"Page numbers must be positive integers (>= 1); got {p}."
                )

        # Deduplicate and sort
        deduped = sorted(set(v))
        if len(deduped) < len(v):
            logger.debug("Deduplicated page list from %s to %s.", v, deduped)
        return deduped


# ---------------------------------------------------------------------------
# GoldQuestion
# ---------------------------------------------------------------------------

# Allowed semantic question categories.
QuestionType = Literal["factoid", "numeric_detail", "multi_chunk", "unanswerable"]


class GoldQuestion(BaseModel):
    """A benchmark question with verified ground-truth evidence.

    Supports all four question categories with cross-field consistency
    enforcement between answerability, evidence, and question type.

    Attributes:
        id: Unique identifier (e.g. 'q001'). Uniqueness across the full
            dataset is enforced by load_gold_dataset, not here.
        type: Semantic category governing evaluation strategy:
            - 'factoid': Single short-answer fact from one page.
            - 'numeric_detail': Numerical value or metric extraction.
            - 'multi_chunk': Answer requires evidence across multiple
              chunks or pages (possibly multiple documents).
            - 'unanswerable': No answer exists in the corpus.
        question: Non-empty natural language query text.
        reference_answer: Authoritative ground-truth answer string.
            For unanswerable questions, this should state why no answer
            is available.
        is_answerable: True when the corpus contains sufficient information
            to answer the question. Must be False for type='unanswerable'.
        gold_evidence: Ordered list of evidence entries. Must be non-empty
            for answerable questions; must be empty for unanswerable ones.
            No two entries may reference the same doc_id.
    """

    id: str
    type: QuestionType
    question: str
    reference_answer: str
    is_answerable: bool
    gold_evidence: List[GoldEvidence]

    # ------------------------------------------------------------------
    # Field validators
    # ------------------------------------------------------------------

    @field_validator("id")
    @classmethod
    def id_not_empty(cls, v: str) -> str:
        """Reject empty question IDs."""
        if not v or not v.strip():
            raise ValueError("Question id must be a non-empty string.")
        return v.strip()

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        """Reject empty or whitespace-only question text."""
        if not v or not v.strip():
            raise ValueError("question must be a non-empty string.")
        return v.strip()

    @field_validator("reference_answer")
    @classmethod
    def reference_answer_not_empty(cls, v: str) -> str:
        """Reject empty reference answers."""
        if not v or not v.strip():
            raise ValueError("reference_answer must be a non-empty string.")
        return v.strip()

    # ------------------------------------------------------------------
    # Cross-field model validators
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def validate_answerability_evidence_consistency(self) -> "GoldQuestion":
        """Enforce cross-field rules between type, is_answerable, and gold_evidence.

        Rules:
        1. type='unanswerable' implies is_answerable must be False.
        2. type='unanswerable' implies gold_evidence must be empty.
        3. is_answerable=False implies gold_evidence must be empty.
        4. is_answerable=True  implies gold_evidence must have >= 1 entry.

        Returns:
            GoldQuestion: The validated model instance.

        Raises:
            ValueError: If any consistency rule is violated.
        """
        is_unanswerable_type = self.type == "unanswerable"

        # Rule 1
        if is_unanswerable_type and self.is_answerable:
            raise ValueError(
                "Questions of type 'unanswerable' must have is_answerable=False."
            )

        # Rules 2 & 3
        if not self.is_answerable and self.gold_evidence:
            raise ValueError(
                f"Question '{self.id}' has is_answerable=False but contains "
                f"{len(self.gold_evidence)} gold_evidence entry(s). "
                "Unanswerable questions must have an empty gold_evidence list."
            )

        # Rule 4
        if self.is_answerable and not self.gold_evidence:
            raise ValueError(
                f"Answerable question '{self.id}' must have at least one "
                "gold_evidence entry."
            )

        return self

    @model_validator(mode="after")
    def validate_no_duplicate_doc_ids(self) -> "GoldQuestion":
        """Reject gold_evidence entries that reference the same doc_id twice.

        Two evidence entries pointing to the same document with different page
        lists are ambiguous. Authors should merge the page lists into a single
        GoldEvidence entry.

        Returns:
            GoldQuestion: The validated model instance.

        Raises:
            ValueError: If any doc_id appears more than once in gold_evidence.
        """
        seen: Dict[str, int] = {}
        for idx, ev in enumerate(self.gold_evidence):
            if ev.doc_id in seen:
                raise ValueError(
                    f"Question '{self.id}' has duplicate gold_evidence for "
                    f"doc_id='{ev.doc_id}' at indices {seen[ev.doc_id]} and {idx}. "
                    "Merge the page lists into a single GoldEvidence entry."
                )
            seen[ev.doc_id] = idx

        return self


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

class DatasetValidationError(Exception):
    """Raised when the gold dataset fails corpus-level validation.

    Collects all violations before raising so the caller sees every
    problem at once rather than failing on the first one.
    """


def build_corpus_registry(papers_dir: Path) -> CorpusRegistry:
    """Build a doc_id -> page_count registry by scanning a PDF directory.

    Uses PyMuPDF to open each PDF and read its page count. Documents that
    cannot be opened are logged as warnings and excluded from the registry.

    Args:
        papers_dir: Directory containing PDF files whose stems are doc_ids.

    Returns:
        CorpusRegistry: Mapping of doc_id strings to integer page counts.

    Raises:
        FileNotFoundError: If papers_dir does not exist.
    """
    import pymupdf  # local import to avoid hard dependency at module level

    papers_dir = Path(papers_dir)
    if not papers_dir.is_dir():
        raise FileNotFoundError(f"Papers directory not found: {papers_dir}")

    registry: CorpusRegistry = {}
    for pdf_path in sorted(papers_dir.glob("*.pdf")):
        doc_id = pdf_path.stem
        try:
            doc = pymupdf.open(pdf_path)
            registry[doc_id] = len(doc)
            doc.close()
        except Exception as exc:
            logger.warning(
                "Could not open '%s' while building corpus registry: %s",
                pdf_path.name,
                exc,
            )

    logger.info(
        "Corpus registry built: %d documents from '%s'.", len(registry), papers_dir
    )
    return registry


def load_gold_dataset(
    dataset_path: Path,
    corpus_registry: Optional[CorpusRegistry] = None,
) -> List[GoldQuestion]:
    """Load, parse, and validate benchmark questions from a JSON file.

    Validation is performed in two passes:

    Pass 1 — Per-model (Pydantic):
        Each JSON object is parsed into a GoldQuestion. Field and model
        validators enforce internal consistency: non-empty strings, page
        number positivity, answerability/evidence alignment, and
        duplicate doc_id detection within a single question.

    Pass 2 — Dataset-level:
        After all questions are parsed, the full list is scanned for:
        - Duplicate question IDs across the dataset.
        - If corpus_registry is provided:
            - Every gold doc_id must be a key in the registry.
            - Every gold page number must be <= the document's page count.

    Args:
        dataset_path: Path to a JSON file containing a list of question
            objects conforming to the GoldQuestion schema.
        corpus_registry: Optional mapping of doc_id -> page_count used for
            corpus-aware validation. When None, corpus checks are skipped
            (useful for offline unit tests with synthetic data).

    Returns:
        List[GoldQuestion]: Validated, ordered list of benchmark questions.

    Raises:
        FileNotFoundError: If dataset_path does not exist.
        ValueError: If the JSON is malformed or not a list.
        DatasetValidationError: If any dataset-level or corpus validation
            rule is violated. The error message lists all violations.
    """
    dataset_path = Path(dataset_path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Gold dataset file not found: {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError(
            f"Gold dataset must be a JSON array; got {type(raw).__name__}."
        )

    # ------------------------------------------------------------------
    # Pass 1: Parse and per-model validate each question
    # ------------------------------------------------------------------
    questions: List[GoldQuestion] = []
    parse_errors: List[str] = []

    for idx, item in enumerate(raw):
        try:
            questions.append(GoldQuestion.model_validate(item))
        except Exception as exc:
            q_id = (
                item.get("id", f"<index {idx}>")
                if isinstance(item, dict)
                else f"<index {idx}>"
            )
            parse_errors.append(f"  [{q_id}] {exc}")

    if parse_errors:
        raise DatasetValidationError(
            f"Failed to parse {len(parse_errors)} question(s):\n"
            + "\n".join(parse_errors)
        )

    # ------------------------------------------------------------------
    # Pass 2: Dataset-level validation
    # ------------------------------------------------------------------
    violations: List[str] = []

    # Rule A: Question IDs must be unique across the dataset
    seen_ids: Dict[str, int] = {}
    for pos, q in enumerate(questions):
        if q.id in seen_ids:
            violations.append(
                f"  Duplicate question ID '{q.id}' at positions "
                f"{seen_ids[q.id]} and {pos}."
            )
        else:
            seen_ids[q.id] = pos

    # Rules B & C: Corpus-aware checks (only when registry is provided)
    if corpus_registry is not None:
        for q in questions:
            for ev in q.gold_evidence:
                # Rule B: doc_id must exist in corpus
                if ev.doc_id not in corpus_registry:
                    violations.append(
                        f"  [{q.id}] doc_id '{ev.doc_id}' does not exist in "
                        "the corpus. Check the PDF directory."
                    )
                    continue  # Cannot validate pages without knowing doc size

                # Rule C: Every page must be within the document's range
                max_page = corpus_registry[ev.doc_id]
                for page in ev.pages:
                    if page > max_page:
                        violations.append(
                            f"  [{q.id}] Page {page} is out of range for "
                            f"'{ev.doc_id}' (document has {max_page} page(s)). "
                            "Page numbers are 1-indexed."
                        )

    if violations:
        raise DatasetValidationError(
            f"Gold dataset failed validation with {len(violations)} violation(s):\n"
            + "\n".join(violations)
        )

    logger.info(
        "Gold dataset loaded: %d questions from '%s'.", len(questions), dataset_path
    )
    return questions
