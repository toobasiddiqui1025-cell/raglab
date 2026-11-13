"""Language model interface and client implementations.

Provides an abstract LLMClient interface and implementations including GeminiClient
(via google-genai with retry logic) and MockLLMClient for offline automated testing.
Allows swapping model providers (e.g., Claude, OpenAI) without altering pipeline logic.
"""

import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)

GEMINI_API_KEY_ENV = "GEMINI_API_KEY"


class LLMError(Exception):
    """Base class for provider-independent LLM client failures."""


class LLMConfigurationError(LLMError):
    """Raised when a client is misconfigured (e.g. missing API key)."""


class LLMAuthenticationError(LLMError):
    """Raised when the provider rejects the credentials or access (401/403)."""


class LLMResponseError(LLMError):
    """Raised when the provider returns no usable text (blocked, empty, truncated)."""


class LLMResponse(BaseModel):
    """Encapsulates generated text output along with token usage metrics.

    Attributes:
        text: Generated completion string.
        prompt_tokens: Number of tokens in the input prompt.
        completion_tokens: Number of tokens generated in the response.
        total_tokens: Total tokens consumed.
    """
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LLMClient(ABC):
    """Abstract base class for Large Language Model clients."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """Generate response text for a given prompt.

        Args:
            prompt: User-facing prompt or assembled RAG context prompt.
            system_instruction: Optional system instruction steering model behavior.
            temperature: Sampling temperature (0.0 for deterministic output).

        Returns:
            LLMResponse: Structured response with output text and token counts.
        """
        pass


class GeminiClient(LLMClient):
    """LLM client backed by the Google Gemini API (google-genai SDK).

    The API key is read from the GEMINI_API_KEY environment variable (or passed
    explicitly) and is never logged. Transient failures (HTTP 429, 5xx, network
    timeouts) are retried with exponential backoff; other failures are raised
    immediately as LLMError subclasses so callers can handle them without
    depending on SDK types.

    Text generation only: this uses a Gemini generative model, never an
    embedding model.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash",
        *,
        max_output_tokens: Optional[int] = None,
        json_mode: bool = False,
        timeout_seconds: float = 60.0,
        max_retries: int = 3,
        backoff_base_seconds: float = 1.0,
        client: Any = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Initialize the Gemini client.

        Args:
            api_key: Gemini API key. Defaults to the GEMINI_API_KEY environment variable.
            model_name: Target Gemini model identifier.
            max_output_tokens: Optional cap on generated tokens.
            json_mode: If True, ask Gemini to return application/json output.
            timeout_seconds: Per-request timeout.
            max_retries: Retries after the first attempt for transient errors.
            backoff_base_seconds: Base delay; doubles each retry (capped at 30s).
            client: Pre-built google.genai.Client (used for testing/injection).
            sleep: Sleep function, injectable so tests don't wait.

        Raises:
            LLMConfigurationError: If no API key is available.
        """
        self.model_name = model_name
        self.max_output_tokens = max_output_tokens
        self.json_mode = json_mode
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.backoff_base_seconds = backoff_base_seconds
        self._sleep = sleep

        if client is not None:
            self._client = client
            return

        key = api_key or os.environ.get(GEMINI_API_KEY_ENV)
        if not key:
            raise LLMConfigurationError(
                f"No Gemini API key found. Set the {GEMINI_API_KEY_ENV} environment "
                "variable (see .env.example) or pass api_key."
            )
        from google import genai
        from google.genai import types

        self._client = genai.Client(
            api_key=key,
            http_options=types.HttpOptions(timeout=int(timeout_seconds * 1000)),
        )

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """Generate a completion, retrying transient errors with exponential backoff.

        Raises:
            LLMAuthenticationError: On HTTP 401/403.
            LLMResponseError: If the response is blocked, empty, or (in JSON
                mode) truncated.
            LLMError: On any other failure, or when retries are exhausted.
        """
        from google.genai import errors, types

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
            max_output_tokens=self.max_output_tokens,
            response_mime_type="application/json" if self.json_mode else None,
        )

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.models.generate_content(
                    model=self.model_name, contents=prompt, config=config
                )
                return self._to_llm_response(response)
            except errors.APIError as exc:
                code = getattr(exc, "code", None)
                if code in (401, 403):
                    raise LLMAuthenticationError(
                        f"Gemini rejected the request (HTTP {code}); check "
                        f"{GEMINI_API_KEY_ENV} and API access."
                    ) from exc
                if code != 429 and not (isinstance(code, int) and code >= 500):
                    raise LLMError(f"Gemini API error (HTTP {code}): {exc}") from exc
                last_error = exc
            except Exception as exc:
                if isinstance(exc, LLMError) or not _is_transient_network_error(exc):
                    raise
                last_error = exc

            if attempt < self.max_retries:
                delay = min(self.backoff_base_seconds * (2**attempt), 30.0)
                logger.warning(
                    "Transient Gemini error (%s); retry %d/%d in %.1fs.",
                    type(last_error).__name__,
                    attempt + 1,
                    self.max_retries,
                    delay,
                )
                self._sleep(delay)

        raise LLMError(
            f"Gemini request failed after {self.max_retries + 1} attempt(s): {last_error}"
        ) from last_error

    def _to_llm_response(self, response: Any) -> LLMResponse:
        """Convert an SDK response into LLMResponse, rejecting unusable output."""
        text = getattr(response, "text", None)
        if not text or not text.strip():
            feedback = getattr(response, "prompt_feedback", None)
            block = getattr(feedback, "block_reason", None)
            raise LLMResponseError(
                "Gemini returned no text"
                + (f" (prompt blocked: {block})" if block else "")
                + "."
            )
        if self.json_mode and _finish_reason(response) == "MAX_TOKENS":
            raise LLMResponseError(
                "Gemini output was truncated at max_output_tokens; JSON would be "
                "incomplete. Increase max_output_tokens."
            )
        usage = getattr(response, "usage_metadata", None)
        prompt_tokens = getattr(usage, "prompt_token_count", None) or 0
        completion_tokens = getattr(usage, "candidates_token_count", None) or 0
        total_tokens = getattr(usage, "total_token_count", None) or (
            prompt_tokens + completion_tokens
        )
        return LLMResponse(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )


def _finish_reason(response: Any) -> str:
    """Return the first candidate's finish reason name, or ''."""
    candidates = getattr(response, "candidates", None) or []
    reason = getattr(candidates[0], "finish_reason", None) if candidates else None
    return getattr(reason, "name", str(reason)) if reason is not None else ""


def _is_transient_network_error(exc: Exception) -> bool:
    """True for timeouts and connection-level failures worth retrying."""
    import httpx

    return isinstance(exc, (httpx.TimeoutException, httpx.TransportError, TimeoutError, ConnectionError))


class MockLLMClient(LLMClient):
    """Mock LLM client returning deterministic responses for unit and CI testing."""

    def __init__(self, default_response: str = "Mock answer [doc1, p.1].") -> None:
        """Initialize mock client with a default canned response."""
        self.default_response = default_response

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """Return canned mock response without issuing network calls."""
        return LLMResponse(
            text=self.default_response,
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        )
