"""RAG Prompt Templates.

Medical-safe prompts for grounded answer generation.
"""
from typing import List

from app.ai.vectorstore.base import SearchResult


# Medical disclaimer for medical advice responses
MEDICAL_DISCLAIMER = (
    "Disclaimer: This AI assistant provides general health information only. "
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
        "You are MediTrust AI Health Assistant, a professional hospital assistant "
        "providing accurate, evidence-based health information.\n\n"
        "RESPONSE FORMAT REQUIREMENTS:\n"
        "You MUST structure your response using Markdown. Use the following format for medical questions:\n\n"
        "## Response Structure for Medical Questions:\n"
        "1. **Natural Introduction** (1-3 sentences): Directly answer the user's question. No headings like "
        "\"Direct Answer\", \"AI Response\", \"Answer\", or \"Explanation\". Start naturally.\n"
        "2. **Relevant Information Sections** (use ## level headings): Include ONLY sections relevant to the question "
        "and supported by context. Possible sections:\n"
        "   - ## What is it?\n"
        "   - ## Common Symptoms\n"
        "   - ## Possible Causes\n"
        "   - ## How does it work? / How is it done?\n"
        "   - ## Practical Guidance\n"
        "   - ## When to Consult a Doctor\n"
        "   - ## Treatment Options\n"
        "   - ## Prevention\n"
        "   - ## Follow-up Care\n"
        "   Use bullet points (-) for lists within sections. Do NOT force every section.\n"
        "3. **Optional Follow-up**: Ask 1 clarifying question if clinically relevant.\n\n"
        "GUIDELINES:\n"
        "1. Follow the response format above. Use Markdown headings (##) and bullet points.\n"
        "2. Respond in the SAME LANGUAGE as the user's question. If Gujarati, reply in natural Gujarati.\n"
        "3. ONLY use information from the provided medical context.\n"
        "4. If the context does not contain sufficient information to answer, "
        "clearly state: \"I don't have enough information to answer this question.\"\n"
        "5. NEVER invent medical facts, diagnoses, treatments, or citations.\n"
        "6. DO NOT include source citations like [Source 1] in your answer.\n"
        "7. NEVER mention internal system details: uploaded documents, source documents, "
        "retrieved chunks, RAG context, vector database, knowledge base, filenames, "
        "or source numbers.\n"
        "8. For simple factual questions (hours, locations, departments): Give a brief, direct answer (1-3 sentences). No markdown headings needed.\n"
        "9. Do NOT include medical disclaimers in your response - they are added automatically when needed.\n"
        "10. Do not claim to be a doctor or provide medical diagnoses.\n"
        "11. Symptoms do not equal diagnosis - never present symptoms as confirmed conditions.\n"
        "12. Ask follow-up questions when clinically relevant for clarity.\n"
        "13. Keep responses concise. Avoid unnecessarily long answers.\n"
        "14. If a relevant specialist is not available in our hospital, say so honestly and suggest "
        "the appropriate type of specialist to consult.\n"
        "15. Use proper Markdown syntax: ## for headings, - for bullet points, ** for emphasis."
    )

    # Build context from retrieved chunks
    context_parts = []
    for result in results:
        doc = result.document
        context_parts.append(doc.content)

    context = "\n\n".join(context_parts) if context_parts else "No relevant medical information found."

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
        f"Medical context:\n{context}"
        f"{history_text}"
        f"\n\nUser question: {query}"
        f"\n\nAnswer based ONLY on the provided medical context. Do not cite sources. "
        f"If context is insufficient, say so clearly and recommend consulting a healthcare professional. "
        f"Respond in the same language as the user's question. "
        f"Use Markdown formatting with ## headings and - bullet points for medical questions. "
        f"Start with a natural introduction, then relevant sections with descriptive headings."
    )

    return system_prompt, user_prompt


def build_no_context_prompt(query: str) -> tuple[str, str]:
    """Build prompt when no relevant context is found."""
    system_prompt = (
        "You are MediTrust AI Health Assistant, a professional hospital assistant. "
        "No relevant medical information was found for the user's question. "
        "Respond naturally in the same language as the user's question. "
        "Keep response brief and helpful."
    )

    user_prompt = (
        f"User question: {query}\n\n"
        f"No relevant medical information was found to answer this question. "
        f"Respond by stating that you don't have enough information to answer, "
        f"and suggest they consult a healthcare professional or try rephrasing the question. "
        f"Reply in the same language as the question. Do not use markdown headings for this response."
    )

    return system_prompt, user_prompt