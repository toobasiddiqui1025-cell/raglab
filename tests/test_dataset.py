"""Tests for the gold evaluation dataset schema and loader (evaluation/dataset.py)."""

import json
import tempfile
from pathlib import Path
from typing import List

import pytest

from raglab.evaluation.dataset import (
    CorpusRegistry,
    DatasetValidationError,
    GoldEvidence,
    GoldQuestion,
    load_gold_dataset,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_evidence(doc_id: str = "1706.03762", pages: List[int] = None) -> dict:
    return {"doc_id": doc_id, "pages": pages or [1]}


def _make_question(
    id: str = "q001",
    type: str = "factoid",
    question: str = "What is BERT?",
    reference_answer: str = "A transformer model.",
    is_answerable: bool = True,
    gold_evidence: List[dict] = None,
) -> dict:
    return {
        "id": id,
        "type": type,
        "question": question,
        "reference_answer": reference_answer,
        "is_answerable": is_answerable,
        "gold_evidence": gold_evidence if gold_evidence is not None else [_make_evidence()],
    }


def _write_json(tmp_path: Path, data: object) -> Path:
    p = tmp_path / "gold.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# GoldEvidence — field validation
# ---------------------------------------------------------------------------

class TestGoldEvidence:
    def test_valid_single_page(self) -> None:
        ev = GoldEvidence(doc_id="1706.03762", pages=[7])
        assert ev.doc_id == "1706.03762"
        assert ev.pages == [7]

    def test_valid_multiple_pages_sorted(self) -> None:
        ev = GoldEvidence(doc_id="1810.04805", pages=[3, 1, 2])
        assert ev.pages == [1, 2, 3]  # sorted ascending

    def test_pages_deduplicated(self) -> None:
        ev = GoldEvidence(doc_id="1706.03762", pages=[5, 5, 3, 3, 7])
        assert ev.pages == [3, 5, 7]

    def test_empty_doc_id_rejected(self) -> None:
        with pytest.raises(Exception, match="non-empty"):
            GoldEvidence(doc_id="", pages=[1])

    def test_whitespace_doc_id_rejected(self) -> None:
        with pytest.raises(Exception, match="non-empty"):
            GoldEvidence(doc_id="   ", pages=[1])

    def test_doc_id_stripped(self) -> None:
        ev = GoldEvidence(doc_id="  1706.03762  ", pages=[1])
        assert ev.doc_id == "1706.03762"

    def test_zero_page_rejected(self) -> None:
        with pytest.raises(Exception, match="positive"):
            GoldEvidence(doc_id="1706.03762", pages=[0])

    def test_negative_page_rejected(self) -> None:
        with pytest.raises(Exception, match="positive"):
            GoldEvidence(doc_id="1706.03762", pages=[-1])

    def test_empty_pages_rejected(self) -> None:
        with pytest.raises(Exception, match="at least one"):
            GoldEvidence(doc_id="1706.03762", pages=[])


# ---------------------------------------------------------------------------
# GoldQuestion — field validators
# ---------------------------------------------------------------------------

class TestGoldQuestionFields:
    def test_valid_factoid(self) -> None:
        q = GoldQuestion(**_make_question())
        assert q.id == "q001"
        assert q.type == "factoid"
        assert q.is_answerable is True
        assert len(q.gold_evidence) == 1

    def test_all_valid_types(self) -> None:
        for qtype in ("factoid", "numeric_detail", "multi_chunk", "unanswerable"):
            if qtype == "unanswerable":
                q = GoldQuestion(**_make_question(type=qtype, is_answerable=False, gold_evidence=[]))
            else:
                q = GoldQuestion(**_make_question(type=qtype))
            assert q.type == qtype

    def test_invalid_type_rejected(self) -> None:
        with pytest.raises(Exception):
            GoldQuestion(**_make_question(type="multi_hop"))

    def test_empty_id_rejected(self) -> None:
        with pytest.raises(Exception, match="non-empty"):
            GoldQuestion(**_make_question(id=""))

    def test_empty_question_rejected(self) -> None:
        with pytest.raises(Exception, match="non-empty"):
            GoldQuestion(**_make_question(question="   "))

    def test_empty_reference_answer_rejected(self) -> None:
        with pytest.raises(Exception, match="non-empty"):
            GoldQuestion(**_make_question(reference_answer=""))

    def test_question_stripped(self) -> None:
        q = GoldQuestion(**_make_question(question="  What is BERT?  "))
        assert q.question == "What is BERT?"


# ---------------------------------------------------------------------------
# GoldQuestion — cross-field model validators
# ---------------------------------------------------------------------------

class TestGoldQuestionCrossField:
    def test_valid_unanswerable(self) -> None:
        q = GoldQuestion(**_make_question(
            type="unanswerable",
            is_answerable=False,
            gold_evidence=[],
            reference_answer="This cannot be answered from the provided documents.",
        ))
        assert not q.is_answerable
        assert q.gold_evidence == []

    def test_unanswerable_type_with_is_answerable_true_rejected(self) -> None:
        with pytest.raises(Exception, match="is_answerable=False"):
            GoldQuestion(**_make_question(
                type="unanswerable",
                is_answerable=True,
                gold_evidence=[],
            ))

    def test_unanswerable_type_with_evidence_rejected(self) -> None:
        with pytest.raises(Exception, match="empty gold_evidence"):
            GoldQuestion(**_make_question(
                type="unanswerable",
                is_answerable=False,
                gold_evidence=[_make_evidence()],
            ))

    def test_is_answerable_false_with_evidence_rejected(self) -> None:
        with pytest.raises(Exception, match="empty gold_evidence"):
            GoldQuestion(**_make_question(
                type="factoid",
                is_answerable=False,
                gold_evidence=[_make_evidence()],
            ))

    def test_is_answerable_true_with_no_evidence_rejected(self) -> None:
        with pytest.raises(Exception, match="at least one gold_evidence"):
            GoldQuestion(**_make_question(
                type="factoid",
                is_answerable=True,
                gold_evidence=[],
            ))

    def test_duplicate_doc_id_in_evidence_rejected(self) -> None:
        with pytest.raises(Exception, match="duplicate gold_evidence"):
            GoldQuestion(**_make_question(
                type="multi_chunk",
                is_answerable=True,
                gold_evidence=[
                    _make_evidence("1706.03762", [1]),
                    _make_evidence("1706.03762", [5]),  # duplicate doc_id
                ],
            ))

    def test_multi_chunk_multiple_docs_valid(self) -> None:
        q = GoldQuestion(**_make_question(
            type="multi_chunk",
            is_answerable=True,
            gold_evidence=[
                _make_evidence("1706.03762", [3, 7]),
                _make_evidence("1810.04805", [2]),
            ],
        ))
        assert len(q.gold_evidence) == 2

    def test_numeric_detail_type_valid(self) -> None:
        q = GoldQuestion(**_make_question(type="numeric_detail"))
        assert q.type == "numeric_detail"


# ---------------------------------------------------------------------------
# load_gold_dataset — file loading and Pass 1 parsing
# ---------------------------------------------------------------------------

class TestLoadGoldDataset:
    def test_load_valid_single_question(self, tmp_path: Path) -> None:
        path = _write_json(tmp_path, [_make_question()])
        questions = load_gold_dataset(path)
        assert len(questions) == 1
        assert isinstance(questions[0], GoldQuestion)

    def test_load_multiple_questions(self, tmp_path: Path) -> None:
        data = [_make_question(id=f"q{i:03d}") for i in range(5)]
        path = _write_json(tmp_path, data)
        questions = load_gold_dataset(path)
        assert len(questions) == 5

    def test_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_gold_dataset(tmp_path / "nonexistent.json")

    def test_non_list_json_raises_value_error(self, tmp_path: Path) -> None:
        path = _write_json(tmp_path, {"id": "q001"})
        with pytest.raises(ValueError, match="JSON array"):
            load_gold_dataset(path)

    def test_invalid_question_raises_dataset_error(self, tmp_path: Path) -> None:
        bad = _make_question(question="")  # empty question
        path = _write_json(tmp_path, [bad])
        with pytest.raises(DatasetValidationError, match="Failed to parse"):
            load_gold_dataset(path)

    def test_multiple_errors_reported_together(self, tmp_path: Path) -> None:
        data = [
            _make_question(id="q001", question=""),    # bad question
            _make_question(id="q002", reference_answer=""),  # bad answer
        ]
        path = _write_json(tmp_path, data)
        with pytest.raises(DatasetValidationError) as exc_info:
            load_gold_dataset(path)
        msg = str(exc_info.value)
        assert "2 question(s)" in msg


# ---------------------------------------------------------------------------
# load_gold_dataset — Pass 2: dataset-level validation
# ---------------------------------------------------------------------------

class TestLoadGoldDatasetLevel:
    def test_duplicate_ids_rejected(self, tmp_path: Path) -> None:
        data = [_make_question(id="q001"), _make_question(id="q001")]
        path = _write_json(tmp_path, data)
        with pytest.raises(DatasetValidationError, match="Duplicate question ID"):
            load_gold_dataset(path)

    def test_corpus_valid_doc_id_accepted(self, tmp_path: Path) -> None:
        registry: CorpusRegistry = {"1706.03762": 15}
        data = [_make_question(gold_evidence=[_make_evidence("1706.03762", [7])])]
        path = _write_json(tmp_path, data)
        questions = load_gold_dataset(path, corpus_registry=registry)
        assert len(questions) == 1

    def test_corpus_missing_doc_id_rejected(self, tmp_path: Path) -> None:
        registry: CorpusRegistry = {"1810.04805": 16}  # 1706.03762 not present
        data = [_make_question(gold_evidence=[_make_evidence("1706.03762", [7])])]
        path = _write_json(tmp_path, data)
        with pytest.raises(DatasetValidationError, match="does not exist in the corpus"):
            load_gold_dataset(path, corpus_registry=registry)

    def test_corpus_page_out_of_range_rejected(self, tmp_path: Path) -> None:
        registry: CorpusRegistry = {"1706.03762": 15}
        data = [_make_question(gold_evidence=[_make_evidence("1706.03762", [99])])]
        path = _write_json(tmp_path, data)
        with pytest.raises(DatasetValidationError, match="out of range"):
            load_gold_dataset(path, corpus_registry=registry)

    def test_corpus_page_1_always_valid(self, tmp_path: Path) -> None:
        registry: CorpusRegistry = {"1706.03762": 15}
        data = [_make_question(gold_evidence=[_make_evidence("1706.03762", [1])])]
        path = _write_json(tmp_path, data)
        questions = load_gold_dataset(path, corpus_registry=registry)
        assert questions[0].gold_evidence[0].pages == [1]

    def test_corpus_last_page_valid(self, tmp_path: Path) -> None:
        registry: CorpusRegistry = {"1706.03762": 15}
        data = [_make_question(gold_evidence=[_make_evidence("1706.03762", [15])])]
        path = _write_json(tmp_path, data)
        questions = load_gold_dataset(path, corpus_registry=registry)
        assert questions[0].gold_evidence[0].pages == [15]

    def test_no_corpus_registry_skips_corpus_checks(self, tmp_path: Path) -> None:
        # doc_id that doesn't exist in any real corpus — should still load without registry
        data = [_make_question(gold_evidence=[_make_evidence("fake_doc_xyz", [1])])]
        path = _write_json(tmp_path, data)
        questions = load_gold_dataset(path, corpus_registry=None)
        assert len(questions) == 1

    def test_all_violations_reported_together(self, tmp_path: Path) -> None:
        registry: CorpusRegistry = {"1706.03762": 15}
        data = [
            _make_question(id="q001", gold_evidence=[_make_evidence("bad_doc", [1])]),
            _make_question(id="q002", gold_evidence=[_make_evidence("1706.03762", [999])]),
        ]
        path = _write_json(tmp_path, data)
        with pytest.raises(DatasetValidationError) as exc_info:
            load_gold_dataset(path, corpus_registry=registry)
        msg = str(exc_info.value)
        # Both violations must appear in a single error
        assert "bad_doc" in msg
        assert "999" in msg

    def test_load_actual_gold_questions_file(self) -> None:
        """The committed eval/gold_questions.json must parse cleanly."""
        path = Path("eval/gold_questions.json")
        if not path.is_file():
            pytest.skip("eval/gold_questions.json not found.")
        questions = load_gold_dataset(path)
        assert len(questions) >= 1
        assert all(isinstance(q, GoldQuestion) for q in questions)
