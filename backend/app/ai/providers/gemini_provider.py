"""Google Gemini LLM Provider Implementation.

Uses the official google-genai SDK.
Free tier: https://ai.google.dev/pricing
Models: gemini-3.8-flash, gemini-1.5-flash, gemini-1.5-flash-8b, gemini-1.5-pro
"""
import logging
from typing import List, Optional

from google import genai
from google.genai import types
from google.genai import errors as genai_errors

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
        model: str = "gemini-3.8-flash",
        temperature: float = 0.2,
        max_tokens: int = 2048,
        timeout: int = 60,
        base_url: str = "",  # Not used by Gemini SDK
    ):
        super().__init__(api_key, model, temperature, max_tokens, timeout, base_url)
        self._client = genai.Client(api_key=api_key)

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
            contents.append(types.Content(role="user", parts=[types.Part(text=system_prompt)]))
            contents.append(types.Content(role="model", parts=[types.Part(text="Understood.")]))

        for msg in messages:
            role = "user" if msg.role == "user" else "model"
            contents.append(types.Content(role=role, parts=[types.Part(text=msg.content)]))

        generation_config = types.GenerateContentConfig(
            temperature=temp,
            max_output_tokens=max_toks,
        )

        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=contents,
                config=generation_config,
            )

            if not response.text:
                raise LLMProviderUnavailableError("Empty response from Gemini", self.provider_name)

            usage = None
            if response.usage_metadata:
                usage = {
                    "prompt_tokens": response.usage_metadata.prompt_token_count or 0,
                    "completion_tokens": response.usage_metadata.candidates_token_count or 0,
                }

            finish_reason = None
            if response.candidates:
                finish_reason = str(response.candidates[0].finish_reason)

            return LLMResponse(
                content=response.text,
                model=self.model,
                usage=usage,
                finish_reason=finish_reason,
            )

        except genai_errors.APIError as e:
            # Handle specific error types based on error code/message
            error_msg = str(e)
            if "API_KEY_INVALID" in error_msg or "PERMISSION_DENIED" in error_msg or "401" in error_msg or "403" in error_msg:
                logger.error("Gemini authentication failed: %s", e)
                raise LLMAuthenticationError("Invalid API key or insufficient permissions", self.provider_name, e)
            elif "RESOURCE_EXHAUSTED" in error_msg or "429" in error_msg or "rate limit" in error_msg.lower():
                logger.error("Gemini rate limit exceeded: %s", e)
                raise LLMRateLimitError("Rate limit exceeded", self.provider_name, e)
            elif "DEADLINE_EXCEEDED" in error_msg or "timeout" in error_msg.lower():
                logger.error("Gemini request timeout: %s", e)
                raise LLMTimeoutError("Request timed out", self.provider_name, e)
            elif "NOT_FOUND" in error_msg or "404" in error_msg or "model" in error_msg.lower():
                logger.error("Gemini model not found: %s", e)
                raise LLMModelNotFoundError(f"Model '{self.model}' not found", self.provider_name, e)
            elif "SERVICE_UNAVAILABLE" in error_msg or "503" in error_msg:
                logger.error("Gemini service unavailable: %s", e)
                raise LLMProviderUnavailableError("Service temporarily unavailable", self.provider_name, e)
            else:
                logger.error("Gemini API error: %s", e)
                raise LLMProviderUnavailableError(f"API error: {str(e)}", self.provider_name, e)
        except Exception as e:
            logger.exception("Unexpected Gemini error: %s", e)
            raise LLMProviderUnavailableError(f"Unexpected error: {str(e)}", self.provider_name, e)