"""RAG Chain - Orchestrates retrieval and generation.

Combines retriever, prompt builder, and LLM provider for end-to-end RAG.
"""
import logging
import re
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


# Keywords that indicate medical advice questions (need disclaimer)
MEDICAL_ADVICE_KEYWORDS = [
    "symptom", "diagnosis", "treatment", "medication", "drug", "prescription",
    "disease", "condition", "illness", "pain", "fever", "blood pressure",
    "diabetes", "cancer", "heart", "stroke", "emergency", "urgent",
    "what should i", "what can i", "how to treat", "how to cure",
    "is it normal", "side effect", "dosage", "therapy", "surgery",
    "prognosis", "recovery", "heal", "allergy", "infection", "virus",
    "bacteria", "chronic", "acute", "inflamma", "swelling", "rash",
    "nausea", "vomiting", "dizzy", "headache", "chest pain", "shortness of breath",
]

# Keywords that indicate general hospital info questions (no disclaimer needed)
GENERAL_INFO_KEYWORDS = [
    "visiting hour", "visit hour", "open", "close", "hour", "time",
    "department", "ward", "clinic", "location", "address", "parking",
    "phone", "contact", "email", "website", "direction", "map",
    "cafeteria", "food", "restaurant", "gift shop", "pharmacy",
    "doctor available", "doctor schedule", "appointment", "booking",
    "doctor", "dr.", "dr ", "physician", "provider",
    "insurance", "billing", "payment", "cost", "price", "fee",
    "admission", "discharge", "registration", "check-in", "check-in",
    "patient portal", "medical record", "record", "form",
]


def _is_medical_advice_question(query: str) -> bool:
    """Determine if a question is asking for medical advice (needs disclaimer)."""
    query_lower = query.lower()
    
    # Check for medical advice keywords
    for keyword in MEDICAL_ADVICE_KEYWORDS:
        if keyword in query_lower:
            return True
    
    # Check for general info keywords (these typically DON'T need disclaimer)
    for keyword in GENERAL_INFO_KEYWORDS:
        if keyword in query_lower:
            return False
    
    # Default to True for safety - if uncertain, include disclaimer
    return True


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
            # Add disclaimer for no-context responses (safety first)
            if _is_medical_advice_question(query):
                answer = answer + MEDICAL_DISCLAIMER
            return RAGResponse(
                answer=answer,
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

        # Conditionally add medical disclaimer
        if _is_medical_advice_question(query):
            answer = answer + MEDICAL_DISCLAIMER

        # Format sources (concise)
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
            answer=answer,
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