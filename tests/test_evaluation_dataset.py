"""Unit tests for the gold dataset loader (raglab.evaluation.dataset).

All tests use temporary files and a synthetic corpus registry: no real PDFs,
Chroma database, or external APIs are touched, and the committed
eval/gold_questions.json is never read or modified.
"""

import copy
import json
from pathlib import Path
from typing import Any, Callable, Dict, List

import pytest

from raglab.evaluation.dataset import (
    CorpusRegistry,
    DatasetValidationError,
    GoldQuestion,
    load_gold_dataset,
)


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def registry() -> CorpusRegistry:
    """Synthetic corpus: doc_id -> page count."""
    return {"doc_a": 10, "doc_b": 5}


@pytest.fixture
def valid_records() -> List[Dict[str, Any]]:
    """A small, fully valid dataset covering every question type."""
    return [
        {
            "id": "q001",
            "type": "factoid",
            "question": "What optimizer was used?",
            "reference_answer": "Adam.",
            "is_answerable": True,
            "gold_evidence": [{"doc_id": "doc_a", "pages": [3]}],
        },
        {
            "id": "q002",
            "type": "multi_chunk",
            "question": "How do the two papers compare?",
            "reference_answer": "They differ in scale.",
            "is_answerable": True,
            "gold_evidence": [
                {"doc_id": "doc_a", "pages": [2, 4]},
                {"doc_id": "doc_b", "pages": [1]},
            ],
        },
        {
            "id": "q003",
            "type": "unanswerable",
            "question": "What is the weather today?",
            "reference_answer": "Not covered by the corpus.",
            "is_answerable": False,
            "gold_evidence": [],
        },
    ]


@pytest.fixture
def write_dataset(tmp_path: Path) -> Callable[[Any], Path]:
    """Return a function that writes data as JSON into tmp_path and returns the path."""

    def _write(data: Any) -> Path:
        path = tmp_path / "gold.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    return _write


# ---------------------------------------------------------------------------
# 1. Valid dataset
# ---------------------------------------------------------------------------

def test_load_valid_dataset(write_dataset, valid_records, registry) -> None:
    questions = load_gold_dataset(write_dataset(valid_records), registry)
    assert [q.id for q in questions] == ["q001", "q002", "q003"]
    assert all(isinstance(q, GoldQuestion) for q in questions)
    assert questions[0].gold_evidence[0].doc_id == "doc_a"


def test_load_does_not_modify_input_file(write_dataset, valid_records, registry) -> None:
    path = write_dataset(valid_records)
    before = path.read_bytes()
    load_gold_dataset(path, registry)
    assert path.read_bytes() == before


# ---------------------------------------------------------------------------
# 2. Duplicate question IDs
# ---------------------------------------------------------------------------

def test_duplicate_question_ids_rejected(write_dataset, valid_records, registry) -> None:
    valid_records[1]["id"] = "q001"
    with pytest.raises(DatasetValidationError, match="Duplicate question ID 'q001'"):
        load_gold_dataset(write_dataset(valid_records), registry)


# ---------------------------------------------------------------------------
# 3-5. Field validation: type, question, reference_answer
# ---------------------------------------------------------------------------

def test_invalid_question_type_rejected(write_dataset, valid_records, registry) -> None:
    valid_records[0]["type"] = "essay"
    with pytest.raises(DatasetValidationError) as exc:
        load_gold_dataset(write_dataset(valid_records), registry)
    assert "[q001]" in str(exc.value) and "type" in str(exc.value)


@pytest.mark.parametrize("blank", ["", "   ", "\n\t"])
def test_empty_question_rejected(write_dataset, valid_records, registry, blank) -> None:
    valid_records[0]["question"] = blank
    with pytest.raises(DatasetValidationError) as exc:
        load_gold_dataset(write_dataset(valid_records), registry)
    assert "[q001]" in str(exc.value) and "question" in str(exc.value)


@pytest.mark.parametrize("blank", ["", "   "])
def test_empty_reference_answer_rejected(
    write_dataset, valid_records, registry, blank
) -> None:
    valid_records[0]["reference_answer"] = blank
    with pytest.raises(DatasetValidationError) as exc:
        load_gold_dataset(write_dataset(valid_records), registry)
    assert "[q001]" in str(exc.value) and "reference_answer" in str(exc.value)


# ---------------------------------------------------------------------------
# 6-7. Corpus validation: doc IDs and pages
# ---------------------------------------------------------------------------

def test_nonexistent_doc_id_rejected(write_dataset, valid_records, registry) -> None:
    valid_records[0]["gold_evidence"][0]["doc_id"] = "ghost_doc"
    with pytest.raises(DatasetValidationError) as exc:
        load_gold_dataset(write_dataset(valid_records), registry)
    assert "[q001]" in str(exc.value) and "ghost_doc" in str(exc.value)


def test_page_beyond_document_length_rejected(
    write_dataset, valid_records, registry
) -> None:
    valid_records[0]["gold_evidence"][0]["pages"] = [11]  # doc_a has 10 pages
    with pytest.raises(DatasetValidationError) as exc:
        load_gold_dataset(write_dataset(valid_records), registry)
    assert "[q001]" in str(exc.value) and "Page 11" in str(exc.value)


def test_last_page_of_document_accepted(write_dataset, valid_records, registry) -> None:
    valid_records[0]["gold_evidence"][0]["pages"] = [10]
    assert len(load_gold_dataset(write_dataset(valid_records), registry)) == 3


@pytest.mark.parametrize("bad_page", [0, -1])
def test_non_positive_page_rejected(
    write_dataset, valid_records, registry, bad_page
) -> None:
    valid_records[0]["gold_evidence"][0]["pages"] = [bad_page]
    with pytest.raises(DatasetValidationError, match="pages"):
        load_gold_dataset(write_dataset(valid_records), registry)


# ---------------------------------------------------------------------------
# 8-9. Answerability / evidence consistency
# ---------------------------------------------------------------------------

def test_answerable_without_evidence_rejected(
    write_dataset, valid_records, registry
) -> None:
    valid_records[0]["gold_evidence"] = []
    with pytest.raises(DatasetValidationError) as exc:
        load_gold_dataset(write_dataset(valid_records), registry)
    assert "q001" in str(exc.value) and "gold_evidence" in str(exc.value)


def test_unanswerable_with_evidence_rejected(
    write_dataset, valid_records, registry
) -> None:
    valid_records[2]["gold_evidence"] = [{"doc_id": "doc_a", "pages": [1]}]
    with pytest.raises(DatasetValidationError) as exc:
        load_gold_dataset(write_dataset(valid_records), registry)
    assert "q003" in str(exc.value) and "gold_evidence" in str(exc.value)


def test_unanswerable_type_flagged_answerable_rejected(
    write_dataset, valid_records, registry
) -> None:
    valid_records[2]["is_answerable"] = True
    with pytest.raises(DatasetValidationError, match="unanswerable"):
        load_gold_dataset(write_dataset(valid_records), registry)


# ---------------------------------------------------------------------------
# 10-11. Valid multi-document and multi-page evidence
# ---------------------------------------------------------------------------

def test_multi_document_evidence_accepted(write_dataset, valid_records, registry) -> None:
    questions = load_gold_dataset(write_dataset(valid_records), registry)
    q = next(q for q in questions if q.id == "q002")
    assert {ev.doc_id for ev in q.gold_evidence} == {"doc_a", "doc_b"}


def test_multi_page_evidence_accepted(write_dataset, valid_records, registry) -> None:
    questions = load_gold_dataset(write_dataset(valid_records), registry)
    q = next(q for q in questions if q.id == "q002")
    assert q.gold_evidence[0].pages == [2, 4]


def test_unsorted_duplicate_pages_normalised(
    write_dataset, valid_records, registry
) -> None:
    valid_records[0]["gold_evidence"][0]["pages"] = [5, 2, 5, 2]
    questions = load_gold_dataset(write_dataset(valid_records), registry)
    assert questions[0].gold_evidence[0].pages == [2, 5]


# ---------------------------------------------------------------------------
# 12. Duplicate evidence
# ---------------------------------------------------------------------------

def test_duplicate_doc_id_in_evidence_rejected(
    write_dataset, valid_records, registry
) -> None:
    valid_records[1]["gold_evidence"] = [
        {"doc_id": "doc_a", "pages": [1]},
        {"doc_id": "doc_a", "pages": [2]},
    ]
    with pytest.raises(DatasetValidationError) as exc:
        load_gold_dataset(write_dataset(valid_records), registry)
    assert "q002" in str(exc.value) and "duplicate" in str(exc.value).lower()


# ---------------------------------------------------------------------------
# 13-14. File and JSON handling
# ---------------------------------------------------------------------------

def test_missing_file_raises_file_not_found(tmp_path: Path, registry) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        load_gold_dataset(tmp_path / "does_not_exist.json", registry)


def test_malformed_json_raises_value_error(tmp_path: Path, registry) -> None:
    path = tmp_path / "bad.json"
    path.write_text('[{"id": "q001",', encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_gold_dataset(path, registry)


def test_top_level_object_rejected(write_dataset, registry) -> None:
    with pytest.raises(ValueError, match="JSON array"):
        load_gold_dataset(write_dataset({"id": "q001"}), registry)


# ---------------------------------------------------------------------------
# Extra: missing fields, unknown fields, aggregated errors
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "field",
    ["id", "type", "question", "reference_answer", "is_answerable", "gold_evidence"],
)
def test_missing_required_field_rejected(
    write_dataset, valid_records, registry, field
) -> None:
    del valid_records[0][field]
    with pytest.raises(DatasetValidationError) as exc:
        load_gold_dataset(write_dataset(valid_records), registry)
    assert field in str(exc.value)


def test_unknown_field_rejected(write_dataset, valid_records, registry) -> None:
    valid_records[0]["extra_field"] = 1
    with pytest.raises(DatasetValidationError, match="extra_field"):
        load_gold_dataset(write_dataset(valid_records), registry)


def test_all_violations_reported_together(
    write_dataset, valid_records, registry
) -> None:
    broken = copy.deepcopy(valid_records)
    broken[0]["gold_evidence"][0]["doc_id"] = "ghost_doc"
    broken[1]["gold_evidence"][1]["pages"] = [99]
    with pytest.raises(DatasetValidationError) as exc:
        load_gold_dataset(write_dataset(broken), registry)
    msg = str(exc.value)
    assert "ghost_doc" in msg and "99" in msg
    assert "2 violation(s)" in msg


def test_corpus_checks_skipped_without_registry(write_dataset, valid_records) -> None:
    valid_records[0]["gold_evidence"][0]["doc_id"] = "anything"
    assert len(load_gold_dataset(write_dataset(valid_records))) == 3


# ---------------------------------------------------------------------------
# Strict typing: no silent coercion of gold labels
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_pages", [[True], ["3"], [3.0], [3.5]])
def test_non_integer_pages_rejected(write_dataset, valid_records, registry, bad_pages) -> None:
    valid_records[0]["gold_evidence"][0]["pages"] = bad_pages
    with pytest.raises(DatasetValidationError, match="gold_evidence.0.pages"):
        load_gold_dataset(write_dataset(valid_records), registry)


@pytest.mark.parametrize("bad_flag", ["yes", "true", 1, 0, None])
def test_non_boolean_is_answerable_rejected(
    write_dataset, valid_records, registry, bad_flag
) -> None:
    valid_records[0]["is_answerable"] = bad_flag
    with pytest.raises(DatasetValidationError, match="is_answerable"):
        load_gold_dataset(write_dataset(valid_records), registry)
