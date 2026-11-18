"""LangSmith tracing configuration and safe input/output processors.

These functions ensure no PHI, patient queries, medical answers, conversation history,
prompts, retrieved document text, or patient identifiers are sent to LangSmith.
Only safe metadata is included in traces.

CRITICAL: These callbacks are invoked by the langsmith SDK with specific signatures:
- process_inputs(inputs_dict: dict) -> dict: receives bound function arguments as a single dict
- process_outputs(output: Any) -> dict: receives raw function return value as single arg

FAIL-CLOSED: On any error, return empty/safe placeholder, NEVER original data.
"""
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("meditrust.ai.tracing.config")


def _score_summary(scores: List[float]) -> Dict[str, float]:
    """Compute min/max/avg of scores without exposing individual scores."""
    if not scores:
        return {"min": 0.0, "max": 0.0, "avg": 0.0}
    try:
        return {
            "min": round(min(scores), 4),
            "max": round(max(scores), 4),
            "avg": round(sum(scores) / len(scores), 4),
        }
    except Exception:
        return {"min": 0.0, "max": 0.0, "avg": 0.0}


def _safe_get(obj: Any, attr: str, default: Any = None) -> Any:
    """Safely get attribute with default."""
    try:
        return getattr(obj, attr, default)
    except Exception:
        return default


def _safe_dict_get(d: Any, key: str, default: Any = None) -> Any:
    """Safely get dict key with default."""
    try:
        if isinstance(d, dict):
            return d.get(key, default)
        return default
    except Exception:
        return default


def _safe_len(obj: Any, default: int = 0) -> int:
    """Safely get length with default."""
    try:
        return len(obj) if obj is not None else default
    except Exception:
        return default


# =============================================================================
# RAG Chain Sanitizers
# =============================================================================

def rag_chain_process_inputs(inputs_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Extract only safe metadata from RAG chain inputs.

    Args:
        inputs_dict: Dict with keys: query, conversation_history, top_k, db

    Returns:
        Dict with only safe metadata.
    """
    # FAIL-CLOSED: Return empty dict on any error
    try:
        if not isinstance(inputs_dict, dict):
            logger.warning("rag_chain_process_inputs: inputs_dict is not a dict")
            return {"top_k": None, "has_conversation_history": False, "history_length": 0}

        top_k = _safe_dict_get(inputs_dict, "top_k")
        conversation_history = inputs_dict.get("conversation_history")

        return {
            "top_k": top_k if isinstance(top_k, int) else None,
            "has_conversation_history": bool(conversation_history),
            "history_length": _safe_len(conversation_history),
        }
    except Exception as e:
        logger.warning("rag_chain_process_inputs failed: %s", type(e).__name__)
        return {"top_k": None, "has_conversation_history": False, "history_length": 0}


def rag_chain_process_outputs(output: Any) -> Dict[str, Any]:
    """Extract only safe metadata from RAG chain outputs.

    Args:
        output: RAGResponse object or None

    Returns:
        Dict with only safe metadata.
    """
    # FAIL-CLOSED: Return minimal safe dict on any error
    try:
        if output is None:
            return {"has_context": False, "sources_count": 0}

        # Handle both RAGResponse and dict-like objects
        has_context = _safe_get(output, "has_context", False)
        sources = _safe_get(output, "sources", [])
        model = _safe_get(output, "model", "")
        doctor_recs = _safe_get(output, "doctor_recommendations", None)
        disclaimer = _safe_get(output, "disclaimer", "")

        scores = []
        if isinstance(sources, list):
            for s in sources:
                if isinstance(s, dict):
                    score = s.get("score")
                    if isinstance(score, (int, float)):
                        scores.append(float(score))

        return {
            "has_context": bool(has_context),
            "sources_count": _safe_len(sources),
            "score_summary": _score_summary(scores),
            "model": str(model) if model else "",
            "has_doctor_recommendations": doctor_recs is not None and _safe_len(doctor_recs) > 0,
            "has_disclaimer": bool(disclaimer),
        }
    except Exception as e:
        logger.warning("rag_chain_process_outputs failed: %s", type(e).__name__)
        return {"has_context": False, "sources_count": 0}


# =============================================================================
# Retriever Sanitizers
# =============================================================================

def retriever_process_inputs(inputs_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Extract only safe metadata from retriever inputs.

    Args:
        inputs_dict: Dict with keys: query, top_k, filter_metadata

    Returns:
        Dict with only safe metadata.
    """
    try:
        if not isinstance(inputs_dict, dict):
            logger.warning("retriever_process_inputs: inputs_dict is not a dict")
            return {"top_k": None, "has_filter": False}

        top_k = _safe_dict_get(inputs_dict, "top_k")
        filter_metadata = inputs_dict.get("filter_metadata")

        return {
            "top_k": top_k if isinstance(top_k, int) else None,
            "has_filter": bool(filter_metadata),
        }
    except Exception as e:
        logger.warning("retriever_process_inputs failed: %s", type(e).__name__)
        return {"top_k": None, "has_filter": False}


def retriever_process_outputs(output: Any) -> Dict[str, Any]:
    """Extract only safe metadata from retriever outputs.

    Args:
        output: List of SearchResult objects

    Returns:
        Dict with only safe metadata.
    """
    # FAIL-CLOSED: Only process list/tuple of results
    try:
        if not isinstance(output, (list, tuple)):
            return {"results_count": 0, "score_summary": _score_summary([])}

        if not output:
            return {"results_count": 0, "score_summary": _score_summary([])}

        results_count = len(output)
        scores = []

        for r in output:
            score = _safe_get(r, "score")
            if isinstance(score, (int, float)):
                scores.append(float(score))

        return {
            "results_count": results_count,
            "score_summary": _score_summary(scores),
        }
    except Exception as e:
        logger.warning("retriever_process_outputs failed: %s", type(e).__name__)
        return {"results_count": 0, "score_summary": _score_summary([])}


# =============================================================================
# FAISS Search Sanitizers
# =============================================================================

def faiss_search_process_inputs(inputs_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Extract only safe metadata from FAISS search inputs.

    Args:
        inputs_dict: Dict with keys: query_embedding, top_k, filter_metadata, self

    Returns:
        Dict with only safe metadata.
    """
    try:
        if not isinstance(inputs_dict, dict):
            logger.warning("faiss_search_process_inputs: inputs_dict is not a dict")
            return {"top_k": None, "embedding_dim": 0, "has_filter": False, "index_size": 0}

        top_k = _safe_dict_get(inputs_dict, "top_k")
        query_embedding = inputs_dict.get("query_embedding")
        filter_metadata = inputs_dict.get("filter_metadata")
        self_obj = inputs_dict.get("self")

        embedding_dim = 0
        if isinstance(query_embedding, (list, tuple)):
            embedding_dim = len(query_embedding)

        index_size = 0
        if self_obj and hasattr(self_obj, "_index") and self_obj._index:
            index_size = _safe_get(self_obj._index, "ntotal", 0)

        return {
            "top_k": top_k if isinstance(top_k, int) else None,
            "embedding_dim": embedding_dim,
            "has_filter": bool(filter_metadata),
            "index_size": index_size if isinstance(index_size, int) else 0,
        }
    except Exception as e:
        logger.warning("faiss_search_process_inputs failed: %s", type(e).__name__)
        return {"top_k": None, "embedding_dim": 0, "has_filter": False, "index_size": 0}


def faiss_search_process_outputs(output: Any) -> Dict[str, Any]:
    """Extract only safe metadata from FAISS search outputs.

    Args:
        output: List of SearchResult objects

    Returns:
        Dict with only safe metadata.
    """
    # FAIL-CLOSED: Only process list/tuple of results
    try:
        if not isinstance(output, (list, tuple)):
            return {"results_count": 0, "score_summary": _score_summary([])}

        if not output:
            return {"results_count": 0, "score_summary": _score_summary([])}

        results_count = len(output)
        scores = []

        for r in output:
            score = _safe_get(r, "score")
            if isinstance(score, (int, float)):
                scores.append(float(score))

        return {
            "results_count": results_count,
            "score_summary": _score_summary(scores),
        }
    except Exception as e:
        logger.warning("faiss_search_process_outputs failed: %s", type(e).__name__)
        return {"results_count": 0, "score_summary": _score_summary([])}


# =============================================================================
# Groq LLM Sanitizers
# =============================================================================

def groq_llm_process_inputs(inputs_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Extract only safe metadata from Groq LLM inputs.

    Args:
        inputs_dict: Dict with keys: messages, system_prompt, temperature, max_tokens, self

    Returns:
        Dict with only safe metadata.
    """
    try:
        if not isinstance(inputs_dict, dict):
            logger.warning("groq_llm_process_inputs: inputs_dict is not a dict")
            return {
                "model": "", "temperature": None, "max_tokens": None,
                "message_count": 0, "has_system_prompt": False
            }

        self_obj = inputs_dict.get("self")
        messages = inputs_dict.get("messages")
        system_prompt = inputs_dict.get("system_prompt")
        temperature = inputs_dict.get("temperature")
        max_tokens = inputs_dict.get("max_tokens")

        model = _safe_get(self_obj, "model", "")
        temp = temperature if temperature is not None else _safe_get(self_obj, "temperature", 0.2)
        max_toks = max_tokens if max_tokens is not None else _safe_get(self_obj, "max_tokens", 2048)

        return {
            "model": str(model) if model else "",
            "temperature": float(temp) if isinstance(temp, (int, float)) else 0.2,
            "max_tokens": int(max_toks) if isinstance(max_toks, int) else 2048,
            "message_count": _safe_len(messages),
            "has_system_prompt": bool(system_prompt),
        }
    except Exception as e:
        logger.warning("groq_llm_process_inputs failed: %s", type(e).__name__)
        return {
            "model": "", "temperature": 0.2, "max_tokens": 2048,
            "message_count": 0, "has_system_prompt": False
        }


def groq_llm_process_outputs(output: Any) -> Dict[str, Any]:
    """Extract only safe metadata from Groq LLM outputs.

    Args:
        output: LLMResponse object

    Returns:
        Dict with only safe metadata.
    """
    try:
        if output is None:
            return {}

        model = _safe_get(output, "model", "")
        finish_reason = _safe_get(output, "finish_reason", "")
        usage = _safe_get(output, "usage", None)

        usage_dict = {}
        if isinstance(usage, dict):
            pt = usage.get("prompt_tokens")
            ct = usage.get("completion_tokens")
            if isinstance(pt, int):
                usage_dict["prompt_tokens"] = pt
            if isinstance(ct, int):
                usage_dict["completion_tokens"] = ct

        return {
            "model": str(model) if model else "",
            "finish_reason": str(finish_reason) if finish_reason else "",
            "usage": usage_dict,
        }
    except Exception as e:
        logger.warning("groq_llm_process_outputs failed: %s", type(e).__name__)
        return {}