"""Tests for scripts/draft_questions.py using mocked LLM clients only.

No real API calls are made, and the real eval/gold_questions.json is never
written (all outputs go to tmp_path).
"""

import importlib.util
import json
import logging
from pathlib import Path
from typing import List, Optional

import pytest

from raglab.generation.llm import LLMClient, LLMResponse, MockLLMClient
from raglab.ingestion.loader import Page

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("draft_questions", ROOT / "scripts" / "draft_questions.py")
dq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dq)

CONFIG = ROOT / "configs" / "baseline.yaml"
LONG = "word " * 200  # > MIN_PAGE_CHARS


def _pages() -> List[Page]:
    out = []
    for doc in ("doc_a", "doc_b", "doc_c"):
        for n in range(1, 5):
            out.append(Page(doc_id=doc, title=doc, page_number=n, text=f"{doc} {n} {LONG}"))
    out.append(Page(doc_id="doc_a", title="doc_a", page_number=5, text="short"))
    return out


def _resp(q_type: str, evidence: list, **over) -> str:
    item = {
        "question": "What is X?",
        "type": q_type,
        "reference_answer": "X is Y.",
        "evidence": evidence,
        "reason": "The page states it.",
    }
    item.update(over)
    return json.dumps({"questions": [item]})


class ScriptedClient(LLMClient):
    """Returns queued responses (or raises queued exceptions)."""

    def __init__(self, *items) -> None:
        self.items = list(items)
        self.calls = 0

    def generate(self, prompt, system_instruction=None, temperature=0.0) -> LLMResponse:
        self.calls += 1
        item = self.items.pop(0) if len(self.items) > 1 else self.items[0]
        if isinstance(item, Exception):
            raise item
        return LLMResponse(text=item)


class EchoClient(LLMClient):
    """Builds a valid response from the page headers found in the prompt."""

    def generate(self, prompt, system_instruction=None, temperature=0.0) -> LLMResponse:
        import re

        refs = re.findall(r'<source_page doc_id="([^"]+)" page="(\d+)">', prompt)
        q_type = re.search(r'Requested type: "(\w+)"', prompt).group(1)
        ev: dict = {}
        for d, p in refs:
            ev.setdefault(d, []).append(int(p))
        evidence = [{"doc_id": d, "pages": ps} for d, ps in ev.items()]
        return LLMResponse(text=_resp(q_type, evidence))


# ---------------------------------------------------------------- sampling

def test_eligible_pages_drops_short_pages() -> None:
    assert all(len(p.text) >= dq.MIN_PAGE_CHARS for p in dq.eligible_pages(_pages()))
    assert len(dq.eligible_pages(_pages())) == 12


def test_sampling_reproducible_with_seed() -> None:
    import random

    pages = dq.eligible_pages(_pages())
    a = dq.sample_single_pages(pages, 4, random.Random(7))
    b = dq.sample_single_pages(pages, 4, random.Random(7))
    c = dq.sample_single_pages(pages, 4, random.Random(8))
    assert [(p.doc_id, p.page_number) for p in a] == [(p.doc_id, p.page_number) for p in b]
    assert [(p.doc_id, p.page_number) for p in a] != [(p.doc_id, p.page_number) for p in c]


def test_page_groups_have_distinct_pages_and_can_span_documents() -> None:
    import random

    groups = dq.sample_page_groups(dq.eligible_pages(_pages()), 20, random.Random(1), cross_doc_probability=0.5)
    assert groups
    for g in groups:
        assert len({(p.doc_id, p.page_number) for p in g}) == len(g) == 2
    assert any(len({p.doc_id for p in g}) > 1 for g in groups)
    assert any(len({p.doc_id for p in g}) == 1 for g in groups)


# ----------------------------------------------------------------- parsing

def test_parse_valid_factoid() -> None:
    page = _pages()[0]
    out = dq.parse_candidates(_resp("factoid", [{"doc_id": "doc_a", "pages": [1]}]), "factoid", [page])
    assert len(out) == 1 and out[0].reason


def test_parse_accepts_markdown_fenced_json() -> None:
    text = "```json\n" + _resp("factoid", [{"doc_id": "doc_a", "pages": [1]}]) + "\n```"
    assert len(dq.parse_candidates(text, "factoid", [_pages()[0]])) == 1


@pytest.mark.parametrize("bad", ["not json at all", "[1, 2]", '{"questions": "nope"}', ""])
def test_parse_malformed_raises_value_error(bad: str) -> None:
    with pytest.raises(ValueError):
        dq.parse_candidates(bad, "factoid", [_pages()[0]])


def test_parse_drops_item_with_missing_reason() -> None:
    data = json.loads(_resp("factoid", [{"doc_id": "doc_a", "pages": [1]}]))
    del data["questions"][0]["reason"]
    assert dq.parse_candidates(json.dumps(data), "factoid", [_pages()[0]]) == []


def test_parse_drops_wrong_type() -> None:
    text = _resp("numeric_detail", [{"doc_id": "doc_a", "pages": [1]}])
    assert dq.parse_candidates(text, "factoid", [_pages()[0]]) == []


def test_parse_drops_hallucinated_page() -> None:
    text = _resp("factoid", [{"doc_id": "doc_a", "pages": [3]}])
    assert dq.parse_candidates(text, "factoid", [_pages()[0]]) == []


def test_parse_drops_multi_chunk_with_single_page() -> None:
    pages = _pages()[:2]
    text = _resp("multi_chunk", [{"doc_id": "doc_a", "pages": [1]}])
    assert dq.parse_candidates(text, "multi_chunk", pages) == []


def test_parse_accepts_multi_document_multi_chunk() -> None:
    pages = [_pages()[0], _pages()[4]]  # doc_a p1, doc_b p1
    text = _resp("multi_chunk", [{"doc_id": "doc_a", "pages": [1]}, {"doc_id": "doc_b", "pages": [1]}])
    assert len(dq.parse_candidates(text, "multi_chunk", pages)) == 1


# ------------------------------------------------------- error handling

def test_api_error_returns_empty_without_raising() -> None:
    client = ScriptedClient(RuntimeError("503 unavailable"))
    out = dq.generate_for_pages(client, "factoid", [_pages()[0]], 1, 0.0, retries=1)
    assert out == [] and client.calls == 2


def test_malformed_then_valid_response_recovers_on_retry() -> None:
    good = _resp("factoid", [{"doc_id": "doc_a", "pages": [1]}])
    client = ScriptedClient("garbage", good)
    out = dq.generate_for_pages(client, "factoid", [_pages()[0]], 1, 0.0, retries=1)
    assert len(out) == 1 and client.calls == 2


def test_unimplemented_client_raises_llm_unavailable() -> None:
    client = ScriptedClient(NotImplementedError())
    with pytest.raises(dq.LLMUnavailableError):
        dq.generate_for_pages(client, "factoid", [_pages()[0]], 1, 0.0)


# ---------------------------------------------------------------- output

def test_gold_dataset_path_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="never be written"):
        dq.assert_safe_output_path(tmp_path / "gold_questions.json")
    with pytest.raises(ValueError):
        dq.write_drafts({}, ROOT / "eval" / "gold_questions.json", force=True)


def test_gold_guard_is_case_insensitive_and_follows_symlinks(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        dq.assert_safe_output_path(Path("eval/Gold_Questions.JSON"))
    link = tmp_path / "innocent.json"
    link.symlink_to(ROOT / "eval" / "gold_questions.json")
    with pytest.raises(ValueError):
        dq.assert_safe_output_path(link)


def test_existing_draft_not_overwritten_without_force(tmp_path: Path) -> None:
    path = tmp_path / "draft.json"
    path.write_text("keep me")
    with pytest.raises(FileExistsError):
        dq.write_drafts({"a": 1}, path)
    assert path.read_text() == "keep me"
    dq.write_drafts({"a": 1}, path, force=True)
    assert json.loads(path.read_text()) == {"a": 1}


def _run_main(tmp_path: Path, monkeypatch, client: Optional[LLMClient], *extra: str) -> int:
    monkeypatch.setattr(dq, "load_directory", lambda _p: _pages())
    out = tmp_path / "drafts" / "gold_questions_draft.json"
    return dq.main(
        ["--config", str(CONFIG), "--output", str(out), "--seed", "3",
         "--factoid", "2", "--numeric", "2", "--multi-chunk", "2", *extra],
        client=client,
    )


def test_main_writes_unverified_drafts_with_warning(tmp_path: Path, monkeypatch, caplog) -> None:
    gold = ROOT / "eval" / "gold_questions.json"
    gold_before = gold.read_bytes()
    with caplog.at_level(logging.INFO):
        assert _run_main(tmp_path, monkeypatch, EchoClient()) == 0
    data = json.loads((tmp_path / "drafts" / "gold_questions_draft.json").read_text())

    assert "UNVERIFIED" in data["WARNING"] and "manually verified" in data["WARNING"]
    assert data["seed"] == 3
    types = {d["type"] for d in data["drafts"]}
    assert types == {"factoid", "numeric_detail", "multi_chunk"}
    for d in data["drafts"]:
        assert d["verified"] is False and d["status"] == "UNVERIFIED_AI_DRAFT"
        assert d["question"] and d["reference_answer"] and d["reason"]
        assert d["gold_evidence"] and all(e["pages"] for e in d["gold_evidence"])
    assert "UNVERIFIED" in caplog.text
    assert gold.read_bytes() == gold_before  # real gold dataset untouched


def test_main_is_reproducible_for_same_seed(tmp_path: Path, monkeypatch) -> None:
    def questions(sub: str) -> list:
        monkeypatch.setattr(dq, "load_directory", lambda _p: _pages())
        out = tmp_path / sub / "d.json"
        dq.main(["--config", str(CONFIG), "--output", str(out), "--seed", "11"], client=EchoClient())
        return [(d["type"], d["gold_evidence"]) for d in json.loads(out.read_text())["drafts"]]

    assert questions("a") == questions("b")


def test_main_refuses_gold_output(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(dq, "load_directory", lambda _p: _pages())
    out = tmp_path / "gold_questions.json"
    assert dq.main(["--config", str(CONFIG), "--output", str(out)], client=EchoClient()) == 1
    assert not out.exists()


def test_main_writes_nothing_when_all_responses_malformed(tmp_path: Path, monkeypatch) -> None:
    assert _run_main(tmp_path, monkeypatch, MockLLMClient("not json")) == 1
    assert not (tmp_path / "drafts").exists()


def test_main_exit_2_when_client_unimplemented(tmp_path: Path, monkeypatch) -> None:
    assert _run_main(tmp_path, monkeypatch, ScriptedClient(NotImplementedError())) == 2


def test_build_llm_client_requires_api_key(monkeypatch) -> None:
    import dotenv

    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    from raglab.config import load_config

    with pytest.raises(dq.LLMUnavailableError, match="GEMINI_API_KEY"):
        dq.build_llm_client(load_config(CONFIG))


def test_dry_run_makes_no_llm_calls_and_writes_nothing(tmp_path: Path, monkeypatch) -> None:
    client = ScriptedClient("unused")
    assert _run_main(tmp_path, monkeypatch, client, "--dry-run") == 0
    assert client.calls == 0 and not (tmp_path / "drafts").exists()
