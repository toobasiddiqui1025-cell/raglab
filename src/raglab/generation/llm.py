"""Language model interface and client implementations.

Provides an abstract LLMClient interface and implementations including GeminiClient
(via google-genai with retry logic) and MockLLMClient for offline automated testing.
Allows swapping model providers (e.g., Claude, OpenAI) without altering pipeline logic.
"""

from abc import ABC, abstractmethod
from typing import Optional
from pydantic import BaseModel


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
    """LLM client implementation backed by the Google Gemini API (google-genai)."""

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.5-flash") -> None:
        """Initialize the Gemini client.

        Args:
            api_key: Gemini API key. Defaults to GEMINI_API_KEY environment variable.
            model_name: Target Gemini model identifier.
        """
        pass

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """Generate completion using Gemini API with exponential backoff on rate limits."""
        raise NotImplementedError


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
