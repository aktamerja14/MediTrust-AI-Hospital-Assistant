"""LLM Provider Factory.

Selects and instantiates the configured LLM provider.
Provider selection is driven by environment variables (LLM_PROVIDER, LLM_MODEL, etc.).
"""
import logging
from typing import Optional

from app.core.config import settings
from app.ai.providers.base import (
    BaseLLMProvider,
    LLMAuthenticationError,
)
from app.ai.providers.gemini_provider import GeminiProvider
from app.ai.providers.groq_provider import GroqProvider
from app.ai.providers.openrouter_provider import OpenRouterProvider
from app.ai.providers.mock_provider import MockProvider

logger = logging.getLogger("meditrust.ai.factory")


class ProviderFactory:
    """Factory for creating LLM provider instances."""

    _PROVIDERS = {
        "gemini": GeminiProvider,
        "groq": GroqProvider,
        "openrouter": OpenRouterProvider,
        "mock": MockProvider,
    }

    _instance: Optional[BaseLLMProvider] = None

    @classmethod
    def create(cls, provider_name: Optional[str] = None) -> BaseLLMProvider:
        """Create and return an LLM provider instance.

        Args:
            provider_name: Override the configured provider (for testing)

        Returns:
            Configured LLM provider instance

        Raises:
            ValueError: If provider is unknown or configuration is invalid
            LLMAuthenticationError: If API key is missing for the provider
        """
        name = (provider_name or settings.LLM_PROVIDER).lower().strip()

        if name not in cls._PROVIDERS:
            available = ", ".join(sorted(cls._PROVIDERS.keys()))
            raise ValueError(
                f"Unknown LLM provider: '{name}'. Available: {available}"
            )

        provider_class = cls._PROVIDERS[name]

        # Mock provider doesn't require API key
        if name != "mock" and not settings.LLM_API_KEY:
            raise LLMAuthenticationError(
                f"LLM_API_KEY is required for provider '{name}'",
                name,
            )

        logger.info("Creating LLM provider: %s (model: %s)", name, settings.LLM_MODEL)

        instance = provider_class(
            api_key=settings.LLM_API_KEY if name != "mock" else "mock-key",
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
            timeout=settings.LLM_TIMEOUT,
            base_url=settings.LLM_BASE_URL,
        )

        cls._instance = instance
        return instance

    @classmethod
    def get_instance(cls) -> BaseLLMProvider:
        """Get the singleton provider instance, creating it if needed."""
        if cls._instance is None:
            return cls.create()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (mainly for testing)."""
        cls._instance = None

    @classmethod
    def list_available(cls) -> list[str]:
        """Return list of supported provider names."""
        return sorted(cls._PROVIDERS.keys())