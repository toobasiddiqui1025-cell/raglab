"""Tests for GeminiClient using mocked SDK responses; no network access."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional

import httpx
import pytest
from google.genai import errors

from raglab.generation.llm import (
    GeminiClient,
    LLMAuthenticationError,
    LLMClient,
    LLMConfigurationError,
    LLMError,
    LLMResponseError,
)
from raglab.ingestion.loader import Page

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "draft_questions.py"
_spec = importlib.util.spec_from_file_location("draft_questions_llm_test", _SCRIPT)
dq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dq)


def _response(text: Any = '{"questions": []}', finish: str = "STOP", usage=(10, 5, 15), block=None):
    return SimpleNamespace(
        text=text,
        candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name=finish))],
        usage_metadata=SimpleNamespace(
            prompt_token_count=usage[0],
            candidates_token_count=usage[1],
            total_token_count=usage[2],
        ),
        prompt_feedback=SimpleNamespace(block_reason=block) if block else None,
    )


class FakeSDK:
    """Stands in for google.genai.Client; plays back queued results/exceptions."""

    def __init__(self, *items: Any) -> None:
        self.items: List[Any] = list(items)
        self.calls: List[dict] = []
        self.models = SimpleNamespace(generate_content=self._gen)

    def _gen(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        item = self.items.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _api_error(code: int) -> errors.APIError:
    cls = errors.ClientError if code < 500 else errors.ServerError
    return cls(code, {"error": {"message": "boom", "status": "X"}}, None)


def _client(fake: FakeSDK, sleeps: Optional[List[float]] = None, **kw: Any) -> GeminiClient:
    sleeps = sleeps if sleeps is not None else []
    return GeminiClient(client=fake, sleep=sleeps.append, **kw)


def _page() -> Page:
    return Page(doc_id="d1", title="t", page_number=2, text="x" * 600)


def test_is_an_llm_client() -> None:
    assert isinstance(_client(FakeSDK()), LLMClient)


def test_generate_returns_text_and_token_usage() -> None:
    fake = FakeSDK(_response("hello"))
    out = _client(fake).generate("p", system_instruction="sys", temperature=0.2)
    assert (out.text, out.prompt_tokens, out.completion_tokens, out.total_tokens) == (
        "hello", 10, 5, 15,
    )
    call = fake.calls[0]
    assert call["model"] == "gemini-2.5-flash" and call["contents"] == "p"
    assert call["config"].system_instruction == "sys"
    assert call["config"].temperature == 0.2


def test_json_mode_requests_json_mime_type() -> None:
    fake = FakeSDK(_response('{"questions": [{"type": "factoid"}]}'))
    out = _client(fake, json_mode=True, max_output_tokens=2048).generate("p")
    cfg = fake.calls[0]["config"]
    assert cfg.response_mime_type == "application/json"
    assert cfg.max_output_tokens == 2048
    assert '"questions"' in out.text


def test_text_mode_does_not_force_json() -> None:
    fake = FakeSDK(_response("x"))
    _client(fake).generate("p")
    assert fake.calls[0]["config"].response_mime_type is None


def test_missing_api_key_raises_configuration_error(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(LLMConfigurationError, match="GEMINI_API_KEY"):
        GeminiClient()


def test_api_key_read_from_environment_and_timeout_applied(monkeypatch) -> None:
    import google.genai as genai

    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    created: dict = {}
    monkeypatch.setattr(genai, "Client", lambda **kw: created.update(kw) or SimpleNamespace())
    GeminiClient(timeout_seconds=7)
    assert created["api_key"] == "test-key-not-real"
    assert created["http_options"].timeout == 7000  # SDK expects milliseconds


def test_retries_transient_errors_with_exponential_backoff() -> None:
    fake = FakeSDK(_api_error(429), _api_error(503), _response("ok"))
    sleeps: List[float] = []
    out = _client(fake, sleeps, backoff_base_seconds=1.0).generate("p")
    assert out.text == "ok" and sleeps == [1.0, 2.0] and len(fake.calls) == 3


def test_retries_exhausted_raises_llm_error() -> None:
    fake = FakeSDK(*[_api_error(503) for _ in range(3)])
    with pytest.raises(LLMError, match="after 3 attempt"):
        _client(fake, max_retries=2).generate("p")
    assert len(fake.calls) == 3


def test_timeout_is_retried() -> None:
    fake = FakeSDK(httpx.ReadTimeout("slow"), _response("ok"))
    assert _client(fake).generate("p").text == "ok"


@pytest.mark.parametrize("code", [401, 403])
def test_auth_errors_not_retried(code: int) -> None:
    fake = FakeSDK(_api_error(code))
    with pytest.raises(LLMAuthenticationError):
        _client(fake).generate("p")
    assert len(fake.calls) == 1


def test_bad_request_not_retried() -> None:
    fake = FakeSDK(_api_error(400))
    with pytest.raises(LLMError, match="HTTP 400"):
        _client(fake).generate("p")
    assert len(fake.calls) == 1


@pytest.mark.parametrize("text", [None, "", "   "])
def test_empty_response_raises_response_error(text) -> None:
    with pytest.raises(LLMResponseError, match="no text"):
        _client(FakeSDK(_response(text))).generate("p")


def test_blocked_prompt_reported() -> None:
    with pytest.raises(LLMResponseError, match="SAFETY"):
        _client(FakeSDK(_response(None, block="SAFETY"))).generate("p")


def test_truncated_json_raises_but_truncated_text_allowed() -> None:
    truncated = _response('{"questions": [{"ty', finish="MAX_TOKENS")
    with pytest.raises(LLMResponseError, match="truncated"):
        _client(FakeSDK(truncated), json_mode=True).generate("p")
    assert _client(FakeSDK(truncated)).generate("p").text.startswith('{"questions"')


def test_missing_usage_metadata_defaults_to_zero() -> None:
    resp = _response("x")
    resp.usage_metadata = None
    assert _client(FakeSDK(resp)).generate("p").total_tokens == 0


def test_output_feeds_draft_script_parser() -> None:
    payload = (
        '{"questions": [{"type": "factoid", "question": "Q?", "reference_answer": "A.", '
        '"evidence": [{"doc_id": "d1", "pages": [2]}], "reason": "R."}]}'
    )
    client = _client(FakeSDK(_response(payload)), json_mode=True)
    out = dq.generate_for_pages(client, "factoid", [_page()], 1, 0.0)
    assert len(out) == 1 and out[0].reason == "R."


def test_auth_failure_is_fatal_for_draft_script() -> None:
    client = _client(FakeSDK(_api_error(403)))
    with pytest.raises(dq.LLMUnavailableError):
        dq.generate_for_pages(client, "factoid", [_page()], 1, 0.0)


def test_empty_response_is_skipped_not_fatal_for_draft_script() -> None:
    client = _client(FakeSDK(_response(None), _response(None)))
    assert dq.generate_for_pages(client, "factoid", [_page()], 1, 0.0, retries=1) == []
