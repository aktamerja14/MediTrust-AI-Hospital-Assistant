"""Embedding Provider Factory.

Selects and instantiates the configured embedding provider.
Provider selection is driven by environment variables (EMBEDDING_PROVIDER, EMBEDDING_MODEL, etc.).
"""
import logging
from typing import Optional

from app.core.config import settings
from app.ai.embeddings.base import BaseEmbeddingProvider
from app.ai.embeddings.local_provider import LocalEmbeddingProvider

logger = logging.getLogger("meditrust.ai.embeddings.factory")


class EmbeddingFactory:
    """Factory for creating embedding provider instances."""

    _PROVIDERS = {
        "local": LocalEmbeddingProvider,
        # "openai": OpenAIEmbeddingProvider,  # Future
        # "gemini": GeminiEmbeddingProvider,  # Future
    }

    _instance: Optional[BaseEmbeddingProvider] = None

    @classmethod
    def create(cls, provider_name: Optional[str] = None) -> BaseEmbeddingProvider:
        """Create and return an embedding provider instance.

        Args:
            provider_name: Override the configured provider (for testing)

        Returns:
            Configured embedding provider instance

        Raises:
            ValueError: If provider is unknown
        """
        name = (provider_name or settings.EMBEDDING_PROVIDER).lower().strip()

        if name not in cls._PROVIDERS:
            available = ", ".join(sorted(cls._PROVIDERS.keys()))
            raise ValueError(
                f"Unknown embedding provider: '{name}'. Available: {available}"
            )

        provider_class = cls._PROVIDERS[name]

        logger.info("Creating embedding provider: %s (model: %s)", name, settings.EMBEDDING_MODEL)

        instance = provider_class(
            model=settings.EMBEDDING_MODEL,
            api_key=settings.EMBEDDING_API_KEY,
        )

        cls._instance = instance
        return instance

    @classmethod
    def get_instance(cls) -> BaseEmbeddingProvider:
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