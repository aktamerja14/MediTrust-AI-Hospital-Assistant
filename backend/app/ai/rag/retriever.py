"""Retriever for RAG Pipeline.

Handles similarity search and context retrieval from vector store.
"""
import logging
from typing import List, Optional, Dict, Any

from app.ai.embeddings.embedding_factory import EmbeddingFactory
from app.ai.vectorstore.factory import VectorStoreFactory
from app.ai.vectorstore.base import SearchResult
from app.core.config import settings

logger = logging.getLogger("meditrust.ai.rag.retriever")


class RetrieverError(Exception):
    """Exception for retrieval errors."""
    pass


class Retriever:
    """Retrieves relevant context for queries."""

    def __init__(
        self,
        top_k: Optional[int] = None,
    ):
        self.top_k = top_k or settings.RAG_TOP_K
        self._embedding_provider = None
        self._vector_store = None

    @property
    def embedding_provider(self):
        if self._embedding_provider is None:
            self._embedding_provider = EmbeddingFactory.get_instance()
        return self._embedding_provider

    @property
    def vector_store(self):
        if self._vector_store is None:
            self._vector_store = VectorStoreFactory.get_instance()
        return self._vector_store

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Retrieve relevant documents for a query.

        Args:
            query: User query
            top_k: Number of results (overrides default)
            filter_metadata: Optional metadata filters

        Returns:
            List of SearchResult sorted by relevance
        """
        k = top_k or self.top_k

        try:
            # Embed query
            query_embedding = self.embedding_provider.embed_query(query)

            # Search vector store
            results = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=k,
                filter_metadata=filter_metadata,
            )

            logger.info("Retrieved %d results for query (top_k=%d)", len(results), k)
            return results

        except Exception as e:
            logger.exception("Retrieval failed for query: %s", query)
            raise RetrieverError(f"Retrieval failed: {str(e)}")

    def retrieve_with_scores(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: float = 0.0,
    ) -> List[SearchResult]:
        """Retrieve and filter by minimum similarity score."""
        results = self.retrieve(query, top_k)
        return [r for r in results if r.score >= min_score]