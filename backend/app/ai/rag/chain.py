"""RAG Chain - Orchestrates retrieval and generation.

Combines retriever, prompt builder, and LLM provider for end-to-end RAG.
"""
import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from app.ai.factory import ProviderFactory
from app.ai.providers.base import LLMMessage
from app.ai.rag.retriever import Retriever, RetrieverError
from app.ai.rag.prompts import build_rag_prompt, build_no_context_prompt, MEDICAL_DISCLAIMER
from app.ai.vectorstore.base import SearchResult
from app.core.config import settings

logger = logging.getLogger("meditrust.ai.rag.chain")


@dataclass
class RAGResponse:
    """Response from RAG chain with sources."""
    answer: str
    sources: List[Dict[str, Any]]
    has_context: bool
    model: str


class RAGChainError(Exception):
    """Exception for RAG chain errors."""
    pass


class RAGChain:
    """End-to-end RAG pipeline."""

    def __init__(self):
        self._retriever = Retriever()
        self._llm_provider = None

    @property
    def llm_provider(self):
        if self._llm_provider is None:
            self._llm_provider = ProviderFactory.get_instance()
        return self._llm_provider

    @property
    def retriever(self):
        return self._retriever

    def generate(
        self,
        query: str,
        conversation_history: Optional[List[dict]] = None,
        top_k: Optional[int] = None,
    ) -> RAGResponse:
        """Generate a grounded answer for the query.

        Args:
            query: User's question
            conversation_history: Previous messages in conversation
            top_k: Number of chunks to retrieve

        Returns:
            RAGResponse with answer, sources, and metadata
        """
        # Retrieve relevant context
        try:
            results = self.retriever.retrieve(query, top_k=top_k)
        except RetrieverError as e:
            logger.error("Retrieval failed: %s", e)
            raise RAGChainError(f"Retrieval failed: {str(e)}")

        # Handle no relevant context
        if not results:
            system_prompt, user_prompt = build_no_context_prompt(query)
            answer = self._call_llm(system_prompt, user_prompt)
            return RAGResponse(
                answer=answer + MEDICAL_DISCLAIMER,
                sources=[],
                has_context=False,
                model=self.llm_provider.model,
            )

        # Build prompt with context
        system_prompt, user_prompt = build_rag_prompt(query, results, conversation_history)

        # Generate answer
        try:
            answer = self._call_llm(system_prompt, user_prompt)
        except Exception as e:
            logger.exception("LLM generation failed: %s", e)
            raise RAGChainError(f"Generation failed: {str(e)}")

        # Format sources
        sources = []
        for result in results:
            doc = result.document
            page_val = doc.metadata.get("page", doc.metadata.get("chunk_index", "N/A"))
            sources.append({
                "document_id": doc.id,
                "source": doc.metadata.get("source", "Unknown"),
                "page": str(page_val),
                "score": round(result.score, 4),
                "content_preview": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
            })

        return RAGResponse(
            answer=answer + MEDICAL_DISCLAIMER,
            sources=sources,
            has_context=True,
            model=self.llm_provider.model,
        )

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Call LLM provider with system and user prompts."""
        messages = [LLMMessage(role="user", content=user_prompt)]
        response = self.llm_provider.generate_response(
            messages=messages,
            system_prompt=system_prompt,
        )
        return response.content