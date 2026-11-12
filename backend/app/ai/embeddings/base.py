"""Base Embedding Provider Interface.

All embedding providers must implement this interface.
Embedding provider is independent of LLM provider.
"""
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np


class EmbeddingProviderError(Exception):
    """Base exception for embedding provider errors."""
    def __init__(self, message: str, provider: str, original_error: Optional[Exception] = None):
        self.provider = provider
        self.original_error = original_error
        super().__init__(f"[{provider}] {message}")


class EmbeddingAuthError(EmbeddingProviderError):
    """Raised when API key is invalid or missing."""
    pass


class EmbeddingRateLimitError(EmbeddingProviderError):
    """Raised when rate limit is exceeded."""
    pass


class EmbeddingTimeoutError(EmbeddingProviderError):
    """Raised when request times out."""
    pass


class EmbeddingModelNotFoundError(EmbeddingProviderError):
    """Raised when model is not found."""
    pass


class BaseEmbeddingProvider(ABC):
    """Abstract base class for all embedding providers."""

    def __init__(
        self,
        model: str,
        api_key: str = "",
        timeout: int = 60,
    ):
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self._dimension: Optional[int] = None
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate provider configuration. Override in subclasses."""
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider identifier (e.g., 'local', 'openai', 'gemini')."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        pass

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of strings to embed

        Returns:
            List of embedding vectors (list of floats)

        Raises:
            EmbeddingAuthError: Invalid API key
            EmbeddingRateLimitError: Rate limit exceeded
            EmbeddingTimeoutError: Request timeout
            EmbeddingModelNotFoundError: Model not found
        """
        pass

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        return self.embed_texts([text])[0]

    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a search query.
        Override if query embedding differs from document embedding.
        """
        return self.embed_text(query)