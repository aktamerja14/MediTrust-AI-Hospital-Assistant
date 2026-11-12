"""Google Gemini LLM Provider Implementation.

Uses the official google-generativeai SDK.
Free tier: https://ai.google.dev/pricing
Models: gemini-1.5-flash, gemini-1.5-flash-8b, gemini-1.5-pro
"""
import logging
from typing import List, Optional

import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

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

logger = logging.getLogger("meditrust.ai.providers.gemini")


class GeminiProvider(BaseLLMProvider):
    """Google Gemini LLM Provider."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-1.5-flash",
        temperature: float = 0.2,
        max_tokens: int = 2048,
        timeout: int = 60,
        base_url: str = "",  # Not used by Gemini SDK
    ):
        super().__init__(api_key, model, temperature, max_tokens, timeout, base_url)
        genai.configure(api_key=api_key)
        self._client = genai.GenerativeModel(model)

    @property
    def provider_name(self) -> str:
        return "gemini"

    def _validate_config(self) -> None:
        super()._validate_config()
        # Gemini model names don't need special validation here
        # The SDK will raise appropriate errors on use

    def generate_response(
        self,
        messages: List[LLMMessage],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Generate response using Gemini API."""
        temp = temperature if temperature is not None else self.temperature
        max_toks = max_tokens if max_tokens is not None else self.max_tokens

        # Prepare content for Gemini (system prompt + user messages)
        contents = []
        if system_prompt:
            contents.append({"role": "user", "parts": [system_prompt]})
            contents.append({"role": "model", "parts": ["Understood."]})

        for msg in messages:
            role = "user" if msg.role == "user" else "model"
            contents.append({"role": role, "parts": [msg.content]})

        generation_config = genai.types.GenerationConfig(
            temperature=temp,
            max_output_tokens=max_toks,
        )

        try:
            response = self._client.generate_content(
                contents,
                generation_config=generation_config,
                request_options={"timeout": self.timeout},
            )

            if not response.text:
                raise LLMProviderUnavailableError("Empty response from Gemini", self.provider_name)

            return LLMResponse(
                content=response.text,
                model=self.model,
                usage={
                    "prompt_tokens": response.usage_metadata.prompt_token_count if response.usage_metadata else 0,
                    "completion_tokens": response.usage_metadata.candidates_token_count if response.usage_metadata else 0,
                } if response.usage_metadata else None,
                finish_reason=str(response.candidates[0].finish_reason) if response.candidates else None,
            )

        except google_exceptions.PermissionDenied as e:
            logger.error("Gemini authentication failed: %s", e)
            raise LLMAuthenticationError("Invalid API key or insufficient permissions", self.provider_name, e)
        except google_exceptions.ResourceExhausted as e:
            logger.error("Gemini rate limit exceeded: %s", e)
            raise LLMRateLimitError("Rate limit exceeded", self.provider_name, e)
        except google_exceptions.DeadlineExceeded as e:
            logger.error("Gemini request timeout: %s", e)
            raise LLMTimeoutError("Request timed out", self.provider_name, e)
        except google_exceptions.NotFound as e:
            logger.error("Gemini model not found: %s", e)
            raise LLMModelNotFoundError(f"Model '{self.model}' not found", self.provider_name, e)
        except google_exceptions.ServiceUnavailable as e:
            logger.error("Gemini service unavailable: %s", e)
            raise LLMProviderUnavailableError("Service temporarily unavailable", self.provider_name, e)
        except google_exceptions.GoogleAPICallError as e:
            logger.error("Gemini API error: %s", e)
            raise LLMProviderUnavailableError(f"API error: {str(e)}", self.provider_name, e)
        except Exception as e:
            logger.exception("Unexpected Gemini error: %s", e)
            raise LLMProviderUnavailableError(f"Unexpected error: {str(e)}", self.provider_name, e)