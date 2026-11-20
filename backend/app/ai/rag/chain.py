"""RAG Chain - Orchestrates retrieval and generation using LangChain LCEL.

Combines retriever, prompt builder, and LLM provider for end-to-end RAG.
"""
import asyncio
import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from app.ai.factory import ProviderFactory
from app.ai.providers.base import LLMMessage
from app.ai.rag.retriever import Retriever, RetrieverError
from app.ai.rag.prompts import build_rag_prompt, build_no_context_prompt, MEDICAL_DISCLAIMER
from app.ai.vectorstore.base import SearchResult
from app.database.connection import db_manager
from app.ai.tracing.config import (
    rag_chain_process_inputs,
    rag_chain_process_outputs,
    retriever_runnable_process_inputs,
    retriever_runnable_process_outputs,
    is_medical_runnable_process_inputs,
    is_medical_runnable_process_outputs,
    detect_specialization_runnable_process_inputs,
    detect_specialization_runnable_process_outputs,
    build_prompt_runnable_process_inputs,
    build_prompt_runnable_process_outputs,
    llm_runnable_process_inputs,
    llm_runnable_process_outputs,
    format_sources_runnable_process_inputs,
    format_sources_runnable_process_outputs,
    doctor_recommendations_runnable_process_inputs,
    doctor_recommendations_runnable_process_outputs,
    final_response_runnable_process_inputs,
    final_response_runnable_process_outputs,
)
from langsmith import traceable, get_current_run_tree
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
    RunnableBranch,
)
from operator import itemgetter

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
    # Use the resolved `database` variable (not db_manager.db directly) for consistency
    if not recommendations:
        general_doc = await database["doctors"].find_one(
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


# =============================================================================
# LCEL Runnable Components
# =============================================================================

def _make_retriever_runnable(retriever: Retriever, min_score: float = MIN_SIMILARITY_SCORE):
    """Create a RunnableLambda for document retrieval with score filtering.
    
    Uses asyncio.to_thread() to avoid blocking the event loop on synchronous
    FAISS retrieval and embedding operations.
    """
    @traceable(
        name="retriever_runnable",
        run_type="retriever",
        process_inputs=retriever_runnable_process_inputs,
        process_outputs=retriever_runnable_process_outputs,
    )
    async def retrieve_fn(inputs: Dict[str, Any]) -> Dict[str, Any]:
        query = inputs["query"]
        top_k = inputs.get("top_k")
        try:
            # Run synchronous retrieval in thread pool to avoid blocking event loop
            results = await asyncio.to_thread(
                retriever.retrieve_with_scores, query, top_k, min_score
            )
            return {**inputs, "results": results}
        except RetrieverError as e:
            logger.error("Retrieval failed: %s", e)
            raise RAGChainError(f"Retrieval failed: {str(e)}")

    return retrieve_fn


def _make_is_medical_runnable():
    """Create a RunnableLambda to determine if query needs medical disclaimer."""
    @traceable(
        name="is_medical_runnable",
        run_type="chain",
        process_inputs=is_medical_runnable_process_inputs,
        process_outputs=is_medical_runnable_process_outputs,
    )
    def is_medical_fn(inputs: Dict[str, Any]) -> Dict[str, Any]:
        query = inputs["query"]
        is_medical = _is_medical_advice_question(query)
        return {**inputs, "is_medical": is_medical}

    return is_medical_fn


def _make_detect_specialization_runnable():
    """Create a RunnableLambda to detect medical specializations from query."""
    @traceable(
        name="detect_specialization_runnable",
        run_type="chain",
        process_inputs=detect_specialization_runnable_process_inputs,
        process_outputs=detect_specialization_runnable_process_outputs,
    )
    def detect_fn(inputs: Dict[str, Any]) -> Dict[str, Any]:
        query = inputs["query"]
        specializations = _detect_specialization(query)
        return {**inputs, "specializations": specializations}

    return detect_fn


def _make_build_prompt_runnable():
    """Create a RunnableLambda to build the appropriate prompt based on retrieval results."""
    @traceable(
        name="build_prompt_runnable",
        run_type="chain",
        process_inputs=build_prompt_runnable_process_inputs,
        process_outputs=build_prompt_runnable_process_outputs,
    )
    def build_prompt_fn(inputs: Dict[str, Any]) -> Dict[str, Any]:
        query = inputs["query"]
        results = inputs.get("results", [])
        conversation_history = inputs.get("conversation_history")

        if not results:
            system_prompt, user_prompt = build_no_context_prompt(query)
            has_context = False
        else:
            system_prompt, user_prompt = build_rag_prompt(query, results, conversation_history)
            has_context = True

        return {
            **inputs,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "has_context": has_context,
        }

    return build_prompt_fn


def _make_llm_runnable(llm_provider):
    """Create a RunnableLambda to call the LLM provider.
    
    Uses asyncio.to_thread() to avoid blocking the event loop on synchronous
    LLM provider API calls.
    """
    @traceable(
        name="llm_runnable",
        run_type="llm",
        process_inputs=llm_runnable_process_inputs,
        process_outputs=llm_runnable_process_outputs,
    )
    async def llm_fn(inputs: Dict[str, Any]) -> Dict[str, Any]:
        system_prompt = inputs["system_prompt"]
        user_prompt = inputs["user_prompt"]

        try:
            messages = [LLMMessage(role="user", content=user_prompt)]
            # Run synchronous LLM call in thread pool to avoid blocking event loop
            response = await asyncio.to_thread(
                llm_provider.generate_response,
                messages=messages,
                system_prompt=system_prompt,
            )
            answer = response.content
        except Exception as e:
            logger.exception("LLM generation failed: %s", e)
            raise RAGChainError(f"Generation failed: {str(e)}")

        return {**inputs, "answer": answer}

    return llm_fn


def _make_format_sources_runnable():
    """Create a RunnableLambda to format source documents."""
    @traceable(
        name="format_sources_runnable",
        run_type="chain",
        process_inputs=format_sources_runnable_process_inputs,
        process_outputs=format_sources_runnable_process_outputs,
    )
    def format_sources_fn(inputs: Dict[str, Any]) -> List[Dict[str, Any]]:
        results = inputs.get("results", [])
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
        return sources

    return format_sources_fn


def _make_doctor_recommendations_runnable():
    """Create a RunnableLambda to fetch doctor recommendations for medical questions."""
    @traceable(
        name="doctor_recommendations_runnable",
        run_type="chain",
        process_inputs=doctor_recommendations_runnable_process_inputs,
        process_outputs=doctor_recommendations_runnable_process_outputs,
    )
    async def doctor_recs_fn(inputs: Dict[str, Any]) -> Dict[str, Any]:
        is_medical = inputs.get("is_medical", False)
        specializations = inputs.get("specializations", [])
        db = inputs.get("db")

        if not is_medical:
            return {**inputs, "doctor_recommendations": [], "disclaimer": ""}

        doctor_recommendations = await _get_doctor_recommendations(specializations, db=db)
        disclaimer = MEDICAL_DISCLAIMER
        return {**inputs, "doctor_recommendations": doctor_recommendations, "disclaimer": disclaimer}

    return doctor_recs_fn


def _make_final_response_runnable(llm_provider):
    """Create a RunnableLambda to construct the final RAGResponse.
    
    Uses pre-captured LangSmith run IDs passed through the chain inputs
    to ensure they refer to the outer RAGChain.generate() execution.
    """
    @traceable(
        name="final_response_runnable",
        run_type="chain",
        process_inputs=final_response_runnable_process_inputs,
        process_outputs=final_response_runnable_process_outputs,
    )
    def final_response_fn(inputs: Dict[str, Any]) -> RAGResponse:
        # Use pre-captured run IDs from the start of generate()
        langsmith_run_id = inputs.get("langsmith_run_id")
        langsmith_trace_id = inputs.get("langsmith_trace_id")

        return RAGResponse(
            answer=inputs["answer"],
            sources=inputs.get("sources", []),
            has_context=inputs.get("has_context", False),
            model=llm_provider.model,
            doctor_recommendations=inputs.get("doctor_recommendations", []),
            disclaimer=inputs.get("disclaimer", ""),
            langsmith_run_id=langsmith_run_id,
            langsmith_trace_id=langsmith_trace_id,
        )

    return final_response_fn


# =============================================================================
# RAG Chain Class with LCEL Pipeline
# =============================================================================

class RAGChain:
    """End-to-end RAG pipeline using LangChain Expression Language (LCEL)."""

    def __init__(self):
        self._retriever = Retriever()
        self._llm_provider = None
        self._chain = None

    @property
    def llm_provider(self):
        if self._llm_provider is None:
            self._llm_provider = ProviderFactory.get_instance()
        return self._llm_provider

    @llm_provider.setter
    def llm_provider(self, value):
        self._llm_provider = value
        self._chain = None  # Invalidate chain cache

    @property
    def retriever(self):
        return self._retriever

    @retriever.setter
    def retriever(self, value):
        self._retriever = value
        self._chain = None  # Invalidate chain cache

    def _build_chain(self):
        """Build the LCEL chain pipeline."""
        # Always rebuild to pick up any mocked retriever/provider
        return self._create_chain()

    def _create_chain(self):
        """Create the LCEL chain pipeline."""
        # Individual runnable components
        retriever_runnable = _make_retriever_runnable(self.retriever)
        is_medical_runnable = _make_is_medical_runnable()
        detect_specialization_runnable = _make_detect_specialization_runnable()
        build_prompt_runnable = _make_build_prompt_runnable()
        llm_runnable = _make_llm_runnable(self.llm_provider)
        format_sources_runnable = _make_format_sources_runnable()
        doctor_recs_runnable = _make_doctor_recommendations_runnable()
        final_response_runnable = _make_final_response_runnable(self.llm_provider)

        # Branch for when we have context
        has_context_branch = (
            RunnablePassthrough.assign(
                sources=format_sources_runnable,
            )
            | doctor_recs_runnable
            | final_response_runnable
        )

        # Branch for when we don't have context
        no_context_branch = (
            RunnablePassthrough.assign(sources=lambda x: [])
            | doctor_recs_runnable
            | final_response_runnable
        )

        # Main chain: retrieve -> detect medical/specialization -> build prompt -> LLM -> branch
        # Include langsmith_run_id and langsmith_trace_id in the parallel pass-through
        # so they reach final_response_runnable for inclusion in the response.
        chain = (
            RunnableParallel({
                "query": itemgetter("query"),
                "conversation_history": itemgetter("conversation_history"),
                "top_k": itemgetter("top_k"),
                "db": itemgetter("db"),
                "langsmith_run_id": itemgetter("langsmith_run_id"),
                "langsmith_trace_id": itemgetter("langsmith_trace_id"),
            })
            | retriever_runnable
            | is_medical_runnable
            | detect_specialization_runnable
            | build_prompt_runnable
            | llm_runnable
            | RunnableBranch(
                (lambda x: x.get("has_context", False), has_context_branch),
                no_context_branch,
            )
        )

        return chain

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
        """Generate a grounded answer for the query using LCEL pipeline.

        Args:
            query: User's question
            conversation_history: Previous messages in conversation
            top_k: Number of chunks to retrieve
            db: Optional database instance for doctor recommendations (uses default if not provided)

        Returns:
            RAGResponse with answer, sources, and metadata
        """
        # Capture LangSmith run IDs at the start of the traced function.
        # The @traceable decorator creates the run tree when this function is invoked,
        # so get_current_run_tree() returns the parent "rag_generate" run tree here.
        run_tree = get_current_run_tree()
        langsmith_run_id = str(run_tree.id) if run_tree else None
        langsmith_trace_id = str(run_tree.trace_id) if run_tree else None

        chain = self._build_chain()

        inputs = {
            "query": query,
            "conversation_history": conversation_history,
            "top_k": top_k,
            "db": db,
            "langsmith_run_id": langsmith_run_id,
            "langsmith_trace_id": langsmith_trace_id,
        }

        try:
            result = await chain.ainvoke(inputs)
            return result
        except RAGChainError:
            raise
        except Exception as e:
            logger.exception("RAG chain execution failed: %s", e)
            raise RAGChainError(f"RAG chain failed: {str(e)}")