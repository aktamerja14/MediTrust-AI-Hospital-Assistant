"""Groq LLM Provider Implementation.

Uses OpenAI-compatible API via openai SDK.
Free tier: https://console.groq.com/
Models: llama-3.1-8b-instant, llama-3.1-70b-versatile, mixtral-8x7b-32768
Base URL: https://api.groq.com/openai/v1
"""
import logging
from typing import List, Optional

from openai import OpenAI
from openai import (
    AuthenticationError as OpenAIAuthError,
    RateLimitError as OpenAIRateLimitError,
    Timeout as OpenAITimeout,
    NotFoundError as OpenAINotFoundError,
    APIConnectionError,
    APIStatusError,
)

from app.ai.providers.base import (
    BaseLLMProvider,
    LLMMessage,
    LLMResponse,
    LLMAuthenticationError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMModelNotFoundError,
    LLMProviderUnavailableError,
)

logger = logging.getLogger("meditrust.ai.providers.groq")


class GroqProvider(BaseLLMProvider):
    """Groq LLM Provider (OpenAI-compatible)."""

    DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"

    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.1-8b-instant",
        temperature: float = 0.2,
        max_tokens: int = 2048,
        timeout: int = 60,
        base_url: str = "",
    ):
        effective_base_url = base_url or self.DEFAULT_BASE_URL
        super().__init__(api_key, model, temperature, max_tokens, timeout, effective_base_url)
        self._client = OpenAI(
            api_key=api_key,
            base_url=effective_base_url,
            timeout=timeout,
        )

    @property
    def provider_name(self) -> str:
        return "groq"

    def generate_response(
        self,
        messages: List[LLMMessage],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Generate response using Groq API (OpenAI-compatible)."""
        temp = temperature if temperature is not None else self.temperature
        max_toks = max_tokens if max_tokens is not None else self.max_tokens

        formatted_messages = self._prepare_messages(messages, system_prompt)

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=formatted_messages,
                temperature=temp,
                max_tokens=max_toks,
            )

            choice = response.choices[0]
            content = choice.message.content or ""

            if not content:
                raise LLMProviderUnavailableError("Empty response from Groq", self.provider_name)

            return LLMResponse(
                content=content,
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                } if response.usage else None,
                finish_reason=choice.finish_reason,
            )

        except OpenAIAuthError as e:
            logger.error("Groq authentication failed: %s", e)
            raise LLMAuthenticationError("Invalid API key", self.provider_name, e)
        except OpenAIRateLimitError as e:
            logger.error("Groq rate limit exceeded: %s", e)
            raise LLMRateLimitError("Rate limit exceeded", self.provider_name, e)
        except OpenAITimeout as e:
            logger.error("Groq request timeout: %s", e)
            raise LLMTimeoutError("Request timed out", self.provider_name, e)
        except OpenAINotFoundError as e:
            logger.error("Groq model not found: %s", e)
            raise LLMModelNotFoundError(f"Model '{self.model}' not found", self.provider_name, e)
        except APIConnectionError as e:
            logger.error("Groq connection error: %s", e)
            raise LLMProviderUnavailableError("Cannot connect to Groq API", self.provider_name, e)
        except APIStatusError as e:
            if e.status_code == 503:
                logger.error("Groq service unavailable: %s", e)
                raise LLMProviderUnavailableError("Service temporarily unavailable", self.provider_name, e)
            logger.error("Groq API error: %s", e)
            raise LLMProviderUnavailableError(f"API error: {e.message}", self.provider_name, e)
        except Exception as e:
            logger.exception("Unexpected Groq error: %s", e)
            raise LLMProviderUnavailableError(f"Unexpected error: {str(e)}", self.provider_name, e)