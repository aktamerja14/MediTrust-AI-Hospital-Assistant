"""RAG Prompt Templates.

Medical-safe prompts for grounded answer generation.
"""
from typing import List

from app.ai.vectorstore.base import SearchResult


# Medical disclaimer for medical advice responses
MEDICAL_DISCLAIMER = (
    "\n\n---\n"
    "**Disclaimer:** This AI assistant provides general health information only. "
    "It is not a substitute for professional medical advice, diagnosis, or treatment. "
    "For medical emergencies, call emergency services immediately. "
    "For personalized medical advice, consult with a qualified healthcare professional."
)


def build_rag_prompt(
    query: str,
    results: List[SearchResult],
    conversation_history: List[dict] = None,
) -> tuple[str, str]:
    """Build system prompt and user prompt for RAG.

    Args:
        query: User's question
        results: Retrieved relevant chunks
        conversation_history: Previous messages in conversation

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    system_prompt = (
        "You are MediTrust AI Health Assistant, a helpful and responsible AI for "
        "health-related questions. Your role is to provide accurate, evidence-based "
        "information grounded in the provided context documents.\n\n"
        "GUIDELINES:\n"
        "1. ONLY use information from the provided context documents.\n"
        "2. If the context does not contain sufficient information to answer, "
        "clearly state: \"I don't have enough information in the provided documents "
        "to answer this question.\"\n"
        "3. NEVER invent medical facts, diagnoses, treatments, or citations.\n"
        "4. Cite sources using [Source Name] format (short).\n"
        "5. Response length guidelines:\n"
        "   - Simple factual questions (hours, locations, departments): 1-3 sentences.\n"
        "   - Complex medical questions: Structured answer with relevant details.\n"
        "   - Unknown information: Brief statement that info is unavailable.\n"
        "6. Do NOT include medical disclaimers in your response - they are added automatically when needed.\n"
        "7. Do not claim to be a doctor or provide medical diagnoses."
    )

    # Build context from retrieved chunks
    context_parts = []
    for i, result in enumerate(results, 1):
        doc = result.document
        source_name = doc.metadata.get("source", "Unknown Document")
        context_parts.append(
            f"[Source {i}: {source_name}]\n{doc.content}"
        )

    context = "\n\n".join(context_parts) if context_parts else "No relevant documents found."

    # Build conversation history
    history_text = ""
    if conversation_history:
        history_parts = []
        for msg in conversation_history[-6:]:  # Last 6 messages
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role and content:
                history_parts.append(f"{role.capitalize()}: {content}")
        if history_parts:
            history_text = "\n\nPrevious conversation:\n" + "\n".join(history_parts)

    user_prompt = (
        f"Context documents:\n{context}"
        f"{history_text}"
        f"\n\nUser question: {query}"
        f"\n\nAnswer based ONLY on the provided context. Cite sources. "
        f"If context is insufficient, say so clearly."
    )

    return system_prompt, user_prompt


def build_no_context_prompt(query: str) -> tuple[str, str]:
    """Build prompt when no relevant context is found."""
    system_prompt = (
        "You are MediTrust AI Health Assistant. The user asked a question but "
        "no relevant documents were found in the knowledge base."
    )

    user_prompt = (
        f"User question: {query}\n\n"
        f"No relevant documents were found in the knowledge base to answer this question. "
        f"Respond by stating that you don't have enough information from the provided "
        f"documents, and suggest they consult a healthcare professional or try "
        f"rephrasing the question."
    )

    return system_prompt, user_prompt