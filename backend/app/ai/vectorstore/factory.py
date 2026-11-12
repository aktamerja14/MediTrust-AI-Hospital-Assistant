"""Vector Store Factory.

Selects and instantiates the configured vector store.
"""
import logging
from typing import Optional

from app.core.config import settings
from app.ai.embeddings.embedding_factory import EmbeddingFactory
from app.ai.vectorstore.base import BaseVectorStore
from app.ai.vectorstore.faiss_store import FAISSVectorStore

logger = logging.getLogger("meditrust.ai.vectorstore.factory")


class VectorStoreFactory:
    """Factory for creating vector store instances."""

    _STORES = {
        "faiss": FAISSVectorStore,
    }

    _instance: Optional[BaseVectorStore] = None

    @classmethod
    def create(cls, store_name: Optional[str] = None) -> BaseVectorStore:
        """Create and return a vector store instance.

        Args:
            store_name: Override the configured store (for testing)

        Returns:
            Configured vector store instance

        Raises:
            ValueError: If store is unknown
        """
        name = (store_name or settings.VECTOR_STORE).lower().strip()

        if name not in cls._STORES:
            available = ", ".join(sorted(cls._STORES.keys()))
            raise ValueError(
                f"Unknown vector store: '{name}'. Available: {available}"
            )

        # Get embedding dimension from embedding provider
        embedding_provider = EmbeddingFactory.get_instance()
        dimension = embedding_provider.dimension

        store_class = cls._STORES[name]

        logger.info("Creating vector store: %s (dimension: %d)", name, dimension)

        instance = store_class(
            dimension=dimension,
            path=settings.VECTOR_STORE_PATH,
        )

        instance.initialize()
        cls._instance = instance
        return instance

    @classmethod
    def get_instance(cls) -> BaseVectorStore:
        """Get the singleton instance, creating it if needed."""
        if cls._instance is None:
            return cls.create()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (mainly for testing)."""
        cls._instance = None

    @classmethod
    def list_available(cls) -> list[str]:
        """Return list of supported store names."""
        return sorted(cls._STORES.keys())