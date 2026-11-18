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
from app.database.connection import db_manager
from app.ai.tracing.config import (
    rag_chain_process_inputs,
    rag_chain_process_outputs,
)
from langsmith import traceable, get_current_run_tree

logger = logging.getLogger("meditrust.ai.rag.chain")


@dataclass
class RAGResponse:
    """Response from RAG chain with sources."""
    answer: str
    sources: List[Dict[str, Any]]
    has_context: bool
    model: str
    doctor_recommendations: List[Dict[str, Any]] = None
    disclaimer: str = ""
    langsmith_run_id: Optional[str] = None
    langsmith_trace_id: Optional[str] = None


class RAGChainError(Exception):
    """Exception for RAG chain errors."""
    pass


# Minimum similarity score threshold for retrieved chunks
# Filters out low-relevance results (e.g., symptom-prediction training data)
# Lowered to 0.3 for local embedding model (all-MiniLM-L6-v2) which produces lower scores
MIN_SIMILARITY_SCORE = 0.3

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

# Medical specialization keywords for doctor matching
SPECIALIZATION_KEYWORDS = {
    "cardiology": ["heart", "cardiac", "cardiology", "chest pain", "blood pressure", "hypertension", "heart attack", "cardiovascular"],
    "neurology": ["brain", "nerve", "neurology", "neurologist", "headache", "migraine", "seizure", "epilepsy", "stroke", "parkinson", "memory", "numbness", "tingling"],
    "dermatology": ["skin", "hair", "nail", "dermatology", "dermatologist", "rash", "acne", "eczema", "psoriasis", "allergy", "itching", "mole", "lesion"],
    "orthopedics": ["bone", "joint", "orthopedic", "orthopedics", "fracture", "arthritis", "knee", "hip", "shoulder", "back pain", "spine", "sports injury", "muscle"],
    "gynecology": ["pregnancy", "gynecology", "gynecologist", "women", "menstrual", "period", "obstetrics", "prenatal", "maternity", "reproductive", "pcod", "pcos"],
    "medicine": ["fever", "cold", "cough", "flu", "general", "physician", "internal medicine", "diabetes", "thyroid", "infection", "viral", "bacterial", "fatigue", "weakness"],
}


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


def _detect_specialization(query: str) -> List[str]:
    """Detect relevant medical specializations from the query."""
    query_lower = query.lower()
    matched_specializations = []
    
    for spec, keywords in SPECIALIZATION_KEYWORDS.items():
        for keyword in keywords:
            if keyword in query_lower:
                matched_specializations.append(spec)
                break
    
    # Default to general medicine if no specific specialization detected
    if not matched_specializations:
        matched_specializations.append("medicine")
    
    return matched_specializations


async def _get_doctor_recommendations(specializations: List[str], limit: int = 2, db=None) -> List[Dict[str, Any]]:
    """Fetch doctor recommendations from database based on specializations."""
    # Use provided db or fall back to db_manager
    database = db if db is not None else db_manager.db
    if database is None:
        return []
    
    recommendations = []
    seen_doctor_ids = set()
    
    for spec in specializations:
        # Build query for matching doctors
        spec_query = {"$or": []}
        if spec == "cardiology":
            spec_query["$or"].append({"specialization": {"$regex": "cardiolog", "$options": "i"}})
        elif spec == "neurology":
            spec_query["$or"].append({"specialization": {"$regex": "neurolog", "$options": "i"}})
        elif spec == "dermatology":
            spec_query["$or"].append({"specialization": {"$regex": "dermatolog", "$options": "i"}})
        elif spec == "orthopedics":
            spec_query["$or"].append({"specialization": {"$regex": "orthoped", "$options": "i"}})
        elif spec == "gynecology":
            spec_query["$or"].append({"specialization": {"$regex": "gynecolog", "$options": "i"}})
        elif spec == "medicine":
            spec_query["$or"].append({"specialization": {"$regex": "general physician|general medicine", "$options": "i"}})
        
        if not spec_query["$or"]:
            continue
            
        cursor = database["doctors"].find(spec_query).limit(limit)
        async for doc in cursor:
            if doc["doctor_id"] not in seen_doctor_ids:
                seen_doctor_ids.add(doc["doctor_id"])
                
                # Determine match reason
                match_reasons = {
                    "cardiology": "Cardiologist for heart-related concerns",
                    "neurology": "Neurologist for neurological concerns and nerve-related symptoms",
                    "dermatology": "Dermatologist for skin, hair, and nail concerns",
                    "orthopedics": "Orthopedics specialist for bone and joint concerns",
                    "gynecology": "Gynecologist for women's health and pregnancy concerns",
                    "medicine": "General physician for common illnesses and fever",
                }
                
                recommendations.append({
                    "doctor_id": doc["doctor_id"],
                    "name": doc["name"],
                    "qualifications": doc["qualifications"],
                    "specialization": doc["specialization"],
                    "experience_years": doc["experience_years"],
                    "photo_path": doc["photo_path"],
                    "phone": doc["phone"],
                    "available_from": doc["available_from"],
                    "available_to": doc["available_to"],
                    "match_reason": match_reasons.get(spec, f"Specialist in {doc['specialization']}"),
                })
                
                if len(recommendations) >= limit:
                    break
        
        if len(recommendations) >= limit:
            break
    
    # If no recommendations found, fall back to general physician
    if not recommendations:
        general_doc = await db_manager.db["doctors"].find_one(
            {"specialization": {"$regex": "general physician|general medicine", "$options": "i"}}
        )
        if general_doc:
            recommendations.append({
                "doctor_id": general_doc["doctor_id"],
                "name": general_doc["name"],
                "qualifications": general_doc["qualifications"],
                "specialization": general_doc["specialization"],
                "experience_years": general_doc["experience_years"],
                "photo_path": general_doc["photo_path"],
                "phone": general_doc["phone"],
                "available_from": general_doc["available_from"],
                "available_to": general_doc["available_to"],
                "match_reason": "General physician for general medical questions",
            })
    
    return recommendations


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

    @traceable(
        name="rag_generate",
        run_type="chain",
        process_inputs=rag_chain_process_inputs,
        process_outputs=rag_chain_process_outputs,
    )
    async def generate(
        self,
        query: str,
        conversation_history: Optional[List[dict]] = None,
        top_k: Optional[int] = None,
        db=None,
    ) -> RAGResponse:
        """Generate a grounded answer for the query.

        Args:
            query: User's question
            conversation_history: Previous messages in conversation
            top_k: Number of chunks to retrieve
            db: Optional database instance for doctor recommendations (uses default if not provided)

        Returns:
            RAGResponse with answer, sources, and metadata
        """
        # Capture LangSmith run IDs at the start of the traced function
        run_tree = get_current_run_tree()
        langsmith_run_id = str(run_tree.id) if run_tree else None
        langsmith_trace_id = str(run_tree.trace_id) if run_tree else None

        # Check if this is a medical question
        is_medical = _is_medical_advice_question(query)
        
        # Retrieve relevant context with minimum similarity threshold
        try:
            results = self.retriever.retrieve_with_scores(query, top_k=top_k, min_score=MIN_SIMILARITY_SCORE)
        except RetrieverError as e:
            logger.error("Retrieval failed: %s", e)
            raise RAGChainError(f"Retrieval failed: {str(e)}")

        # Handle no relevant context
        if not results:
            system_prompt, user_prompt = build_no_context_prompt(query)
            answer = self._call_llm(system_prompt, user_prompt)
            
            # Get doctor recommendations for medical questions
            doctor_recommendations = []
            disclaimer = ""
            if is_medical:
                specializations = _detect_specialization(query)
                doctor_recommendations = await _get_doctor_recommendations(specializations, db=db)
                disclaimer = MEDICAL_DISCLAIMER
            
            return RAGResponse(
                answer=answer,
                sources=[],
                has_context=False,
                model=self.llm_provider.model,
                doctor_recommendations=doctor_recommendations,
                disclaimer=disclaimer,
                langsmith_run_id=langsmith_run_id,
                langsmith_trace_id=langsmith_trace_id,
            )

        # Build prompt with context
        system_prompt, user_prompt = build_rag_prompt(query, results, conversation_history)

        # Generate answer
        try:
            answer = self._call_llm(system_prompt, user_prompt)
        except Exception as e:
            logger.exception("LLM generation failed: %s", e)
            raise RAGChainError(f"Generation failed: {str(e)}")

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

        # Get doctor recommendations for medical questions
        doctor_recommendations = []
        disclaimer = ""
        if is_medical:
            specializations = _detect_specialization(query)
            doctor_recommendations = await _get_doctor_recommendations(specializations, db=db)
            disclaimer = MEDICAL_DISCLAIMER

        return RAGResponse(
            answer=answer,
            sources=sources,
            has_context=True,
            model=self.llm_provider.model,
            doctor_recommendations=doctor_recommendations,
            disclaimer=disclaimer,
            langsmith_run_id=langsmith_run_id,
            langsmith_trace_id=langsmith_trace_id,
        )

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Call LLM provider with system and user prompts."""
        messages = [LLMMessage(role="user", content=user_prompt)]
        response = self.llm_provider.generate_response(
            messages=messages,
            system_prompt=system_prompt,
        )
        return response.content