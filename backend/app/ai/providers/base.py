"""Base LLM Provider Interface.

All LLM providers must implement this interface.
This ensures provider-agnostic chatbot and RAG business logic.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider."""
    content: str
    model: str
    usage: Optional[dict] = None
    finish_reason: Optional[str] = None


@dataclass
class LLMMessage:
    """Standardized message format for all providers."""
    role: str  # "system", "user", "assistant"
    content: str


class LLMProviderError(Exception):
    """Base exception for LLM provider errors."""
    def __init__(self, message: str, provider: str, original_error: Optional[Exception] = None):
        self.provider = provider
        self.original_error = original_error
        super().__init__(f"[{provider}] {message}")


class LLMAuthenticationError(LLMProviderError):
    """Raised when API key is invalid or missing."""
    pass


class LLMRateLimitError(LLMProviderError):
    """Raised when rate limit is exceeded."""
    pass


class LLMTimeoutError(LLMProviderError):
    """Raised when request times out."""
    pass


class LLMModelNotFoundError(LLMProviderError):
    """Raised when model is not found or not accessible."""
    pass


class LLMProviderUnavailableError(LLMProviderError):
    """Raised when provider service is unavailable."""
    pass


class BaseLLMProvider(ABC):
    """Abstract base class for all LLM providers.

    All providers must implement generate_response().
    Provider-specific SDK calls remain inside their respective adapter files.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        timeout: int = 60,
        base_url: str = "",
    ):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.base_url = base_url
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate provider configuration. Override in subclasses."""
        if not self.api_key:
            raise LLMAuthenticationError("API key is required", self.provider_name)

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider identifier (e.g., 'gemini', 'groq', 'openrouter')."""
        pass

    @abstractmethod
    def generate_response(
        self,
        messages: List[LLMMessage],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Generate a response from the LLM.

        Args:
            messages: List of conversation messages
            system_prompt: Optional system prompt to prepend
            temperature: Override default temperature
            max_tokens: Override default max_tokens

        Returns:
            LLMResponse with content, model, usage, and finish_reason

        Raises:
            LLMAuthenticationError: Invalid API key
            LLMRateLimitError: Rate limit exceeded
            LLMTimeoutError: Request timeout
            LLMModelNotFoundError: Model not found
            LLMProviderUnavailableError: Service unavailable
        """
        pass

    def _prepare_messages(
        self,
        messages: List[LLMMessage],
        system_prompt: Optional[str] = None,
    ) -> List[dict]:
        """Convert standardized messages to provider-specific format.

        Override in subclasses if needed.
        """
        formatted = []
        if system_prompt:
            formatted.append({"role": "system", "content": system_prompt})
        formatted.extend([{"role": msg.role, "content": msg.content} for msg in messages])
        return formatted