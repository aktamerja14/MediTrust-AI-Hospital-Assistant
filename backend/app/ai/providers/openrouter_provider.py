"""OpenRouter LLM Provider Implementation.

Uses OpenAI-compatible API via openai SDK.
Free tier: https://openrouter.ai/
Models: Various free models (e.g., meta-llama/llama-3.1-8b-instruct:free)
Base URL: https://openrouter.ai/api/v1
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

logger = logging.getLogger("meditrust.ai.providers.openrouter")


class OpenRouterProvider(BaseLLMProvider):
    """OpenRouter LLM Provider (OpenAI-compatible)."""

    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        api_key: str,
        model: str = "meta-llama/llama-3.1-8b-instruct:free",
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
            default_headers={
                "HTTP-Referer": "https://meditrust.local",
                "X-Title": "MediTrust AI",
            },
        )

    @property
    def provider_name(self) -> str:
        return "openrouter"

    def generate_response(
        self,
        messages: List[LLMMessage],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Generate response using OpenRouter API (OpenAI-compatible)."""
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
                raise LLMProviderUnavailableError("Empty response from OpenRouter", self.provider_name)

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
            logger.error("OpenRouter authentication failed: %s", e)
            raise LLMAuthenticationError("Invalid API key", self.provider_name, e)
        except OpenAIRateLimitError as e:
            logger.error("OpenRouter rate limit exceeded: %s", e)
            raise LLMRateLimitError("Rate limit exceeded", self.provider_name, e)
        except OpenAITimeout as e:
            logger.error("OpenRouter request timeout: %s", e)
            raise LLMTimeoutError("Request timed out", self.provider_name, e)
        except OpenAINotFoundError as e:
            logger.error("OpenRouter model not found: %s", e)
            raise LLMModelNotFoundError(f"Model '{self.model}' not found", self.provider_name, e)
        except APIConnectionError as e:
            logger.error("OpenRouter connection error: %s", e)
            raise LLMProviderUnavailableError("Cannot connect to OpenRouter API", self.provider_name, e)
        except APIStatusError as e:
            if e.status_code == 503:
                logger.error("OpenRouter service unavailable: %s", e)
                raise LLMProviderUnavailableError("Service temporarily unavailable", self.provider_name, e)
            logger.error("OpenRouter API error: %s", e)
            raise LLMProviderUnavailableError(f"API error: {e.message}", self.provider_name, e)
        except Exception as e:
            logger.exception("Unexpected OpenRouter error: %s", e)
            raise LLMProviderUnavailableError(f"Unexpected error: {str(e)}", self.provider_name, e)