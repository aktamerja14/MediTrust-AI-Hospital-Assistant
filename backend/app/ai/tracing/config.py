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
        output: RAGResponse object, dict, or None

    Returns:
        Dict with only safe metadata.
    """
    # FAIL-CLOSED: Return minimal safe dict on any error
    try:
        if output is None:
            return {"has_context": False, "sources_count": 0}

        # Handle both RAGResponse objects and dict outputs
        if isinstance(output, dict):
            has_context = _safe_dict_get(output, "has_context", False)
            sources = _safe_dict_get(output, "sources", [])
            model = _safe_dict_get(output, "model", "")
            doctor_recs = _safe_dict_get(output, "doctor_recommendations", None)
            disclaimer = _safe_dict_get(output, "disclaimer", "")
        else:
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


# =============================================================================
# LCEL Runnable Sanitizers (for internal chain steps)
# =============================================================================

def retriever_runnable_process_inputs(inputs_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Extract only safe metadata from retriever runnable inputs.

    Args:
        inputs_dict: Dict with keys: query, top_k

    Returns:
        Dict with only safe metadata.
    """
    try:
        if not isinstance(inputs_dict, dict):
            return {"top_k": None}

        top_k = _safe_dict_get(inputs_dict, "top_k")
        return {"top_k": top_k if isinstance(top_k, int) else None}
    except Exception as e:
        logger.warning("retriever_runnable_process_inputs failed: %s", type(e).__name__)
        return {"top_k": None}


def retriever_runnable_process_outputs(output: Any) -> Dict[str, Any]:
    """Extract only safe metadata from retriever runnable outputs.

    Args:
        output: Dict with results key containing List[SearchResult]

    Returns:
        Dict with only safe metadata.
    """
    try:
        if not isinstance(output, dict):
            return {"results_count": 0}

        results = output.get("results", [])
        if not isinstance(results, (list, tuple)):
            return {"results_count": 0}

        results_count = len(results)
        scores = []
        for r in results:
            score = _safe_get(r, "score")
            if isinstance(score, (int, float)):
                scores.append(float(score))

        return {
            "results_count": results_count,
            "score_summary": _score_summary(scores),
        }
    except Exception as e:
        logger.warning("retriever_runnable_process_outputs failed: %s", type(e).__name__)
        return {"results_count": 0, "score_summary": _score_summary([])}


def is_medical_runnable_process_inputs(inputs_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Extract only safe metadata from is_medical runnable inputs.

    Args:
        inputs_dict: Dict with keys: query, results, conversation_history, etc.

    Returns:
        Dict with only safe metadata.
    """
    try:
        if not isinstance(inputs_dict, dict):
            return {"has_results": False}

        results = inputs_dict.get("results", [])
        return {"has_results": bool(results)}
    except Exception as e:
        logger.warning("is_medical_runnable_process_inputs failed: %s", type(e).__name__)
        return {"has_results": False}


def is_medical_runnable_process_outputs(output: Any) -> Dict[str, Any]:
    """Extract only safe metadata from is_medical runnable outputs.

    Args:
        output: Dict with is_medical key

    Returns:
        Dict with only safe metadata.
    """
    try:
        if not isinstance(output, dict):
            return {"is_medical": False}

        is_medical = output.get("is_medical", False)
        return {"is_medical": bool(is_medical)}
    except Exception as e:
        logger.warning("is_medical_runnable_process_outputs failed: %s", type(e).__name__)
        return {"is_medical": False}


def detect_specialization_runnable_process_inputs(inputs_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Extract only safe metadata from detect_specialization runnable inputs.

    Args:
        inputs_dict: Dict with keys: query, is_medical, etc.

    Returns:
        Dict with only safe metadata.
    """
    try:
        if not isinstance(inputs_dict, dict):
            return {"is_medical": False}

        is_medical = inputs_dict.get("is_medical", False)
        return {"is_medical": bool(is_medical)}
    except Exception as e:
        logger.warning("detect_specialization_runnable_process_inputs failed: %s", type(e).__name__)
        return {"is_medical": False}


def detect_specialization_runnable_process_outputs(output: Any) -> Dict[str, Any]:
    """Extract only safe metadata from detect_specialization runnable outputs.

    Args:
        output: Dict with specializations key

    Returns:
        Dict with only safe metadata.
    """
    try:
        if not isinstance(output, dict):
            return {"specializations_count": 0}

        specializations = output.get("specializations", [])
        return {"specializations_count": _safe_len(specializations)}
    except Exception as e:
        logger.warning("detect_specialization_runnable_process_outputs failed: %s", type(e).__name__)
        return {"specializations_count": 0}


def build_prompt_runnable_process_inputs(inputs_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Extract only safe metadata from build_prompt runnable inputs.

    Args:
        inputs_dict: Dict with keys: query, results, conversation_history, specializations, is_medical

    Returns:
        Dict with only safe metadata.
    """
    try:
        if not isinstance(inputs_dict, dict):
            return {"has_results": False, "has_history": False}

        results = inputs_dict.get("results", [])
        conversation_history = inputs_dict.get("conversation_history")
        return {
            "has_results": bool(results),
            "results_count": _safe_len(results),
            "has_history": bool(conversation_history),
            "history_length": _safe_len(conversation_history),
        }
    except Exception as e:
        logger.warning("build_prompt_runnable_process_inputs failed: %s", type(e).__name__)
        return {"has_results": False, "has_history": False}


def build_prompt_runnable_process_outputs(output: Any) -> Dict[str, Any]:
    """Extract only safe metadata from build_prompt runnable outputs.

    Args:
        output: Dict with system_prompt, user_prompt, has_context

    Returns:
        Dict with only safe metadata.
    """
    try:
        if not isinstance(output, dict):
            return {"has_context": False}

        has_context = output.get("has_context", False)
        has_system_prompt = bool(output.get("system_prompt"))
        has_user_prompt = bool(output.get("user_prompt"))
        return {
            "has_context": bool(has_context),
            "has_system_prompt": has_system_prompt,
            "has_user_prompt": has_user_prompt,
        }
    except Exception as e:
        logger.warning("build_prompt_runnable_process_outputs failed: %s", type(e).__name__)
        return {"has_context": False}


def llm_runnable_process_inputs(inputs_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Extract only safe metadata from llm runnable inputs.

    Args:
        inputs_dict: Dict with keys: system_prompt, user_prompt

    Returns:
        Dict with only safe metadata.
    """
    try:
        if not isinstance(inputs_dict, dict):
            return {"has_system_prompt": False, "has_user_prompt": False}

        has_system_prompt = bool(inputs_dict.get("system_prompt"))
        has_user_prompt = bool(inputs_dict.get("user_prompt"))
        return {
            "has_system_prompt": has_system_prompt,
            "has_user_prompt": has_user_prompt,
        }
    except Exception as e:
        logger.warning("llm_runnable_process_inputs failed: %s", type(e).__name__)
        return {"has_system_prompt": False, "has_user_prompt": False}


def llm_runnable_process_outputs(output: Any) -> Dict[str, Any]:
    """Extract only safe metadata from llm runnable outputs.

    Args:
        output: Dict with answer key

    Returns:
        Dict with only safe metadata.
    """
    try:
        if not isinstance(output, dict):
            return {"has_answer": False}

        has_answer = bool(output.get("answer"))
        return {"has_answer": has_answer}
    except Exception as e:
        logger.warning("llm_runnable_process_outputs failed: %s", type(e).__name__)
        return {"has_answer": False}


def format_sources_runnable_process_inputs(inputs_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Extract only safe metadata from format_sources runnable inputs.

    Args:
        inputs_dict: Dict with keys: results, etc.

    Returns:
        Dict with only safe metadata.
    """
    try:
        if not isinstance(inputs_dict, dict):
            return {"has_results": False}

        results = inputs_dict.get("results", [])
        return {"has_results": bool(results), "results_count": _safe_len(results)}
    except Exception as e:
        logger.warning("format_sources_runnable_process_inputs failed: %s", type(e).__name__)
        return {"has_results": False}


def format_sources_runnable_process_outputs(output: Any) -> Dict[str, Any]:
    """Extract only safe metadata from format_sources runnable outputs.

    Args:
        output: List of source dicts

    Returns:
        Dict with only safe metadata.
    """
    try:
        if not isinstance(output, (list, tuple)):
            return {"sources_count": 0}

        return {"sources_count": len(output)}
    except Exception as e:
        logger.warning("format_sources_runnable_process_outputs failed: %s", type(e).__name__)
        return {"sources_count": 0}


def doctor_recommendations_runnable_process_inputs(inputs_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Extract only safe metadata from doctor_recommendations runnable inputs.

    Args:
        inputs_dict: Dict with keys: is_medical, specializations, db, has_context

    Returns:
        Dict with only safe metadata.
    """
    try:
        if not isinstance(inputs_dict, dict):
            return {"is_medical": False, "has_specializations": False}

        is_medical = inputs_dict.get("is_medical", False)
        specializations = inputs_dict.get("specializations", [])
        has_context = inputs_dict.get("has_context", False)
        return {
            "is_medical": bool(is_medical),
            "has_specializations": bool(specializations),
            "specializations_count": _safe_len(specializations),
            "has_context": bool(has_context),
        }
    except Exception as e:
        logger.warning("doctor_recommendations_runnable_process_inputs failed: %s", type(e).__name__)
        return {"is_medical": False, "has_specializations": False}


def doctor_recommendations_runnable_process_outputs(output: Any) -> Dict[str, Any]:
    """Extract only safe metadata from doctor_recommendations runnable outputs.

    Args:
        output: Dict with doctor_recommendations, disclaimer

    Returns:
        Dict with only safe metadata.
    """
    try:
        if not isinstance(output, dict):
            return {"has_doctor_recommendations": False, "has_disclaimer": False}

        doctor_recs = output.get("doctor_recommendations", [])
        disclaimer = output.get("disclaimer", "")
        return {
            "has_doctor_recommendations": bool(doctor_recs) and _safe_len(doctor_recs) > 0,
            "doctor_recommendations_count": _safe_len(doctor_recs),
            "has_disclaimer": bool(disclaimer),
        }
    except Exception as e:
        logger.warning("doctor_recommendations_runnable_process_outputs failed: %s", type(e).__name__)
        return {"has_doctor_recommendations": False, "has_disclaimer": False}


def final_response_runnable_process_inputs(inputs_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Extract only safe metadata from final_response runnable inputs.

    Args:
        inputs_dict: Dict with keys: answer, sources, has_context, doctor_recommendations, disclaimer, langsmith_run_id, langsmith_trace_id

    Returns:
        Dict with only safe metadata.
    """
    try:
        if not isinstance(inputs_dict, dict):
            return {"has_answer": False, "has_sources": False}

        has_answer = bool(inputs_dict.get("answer"))
        sources = inputs_dict.get("sources", [])
        has_context = inputs_dict.get("has_context", False)
        doctor_recs = inputs_dict.get("doctor_recommendations", [])
        disclaimer = inputs_dict.get("disclaimer", "")
        has_run_id = bool(inputs_dict.get("langsmith_run_id"))
        has_trace_id = bool(inputs_dict.get("langsmith_trace_id"))
        return {
            "has_answer": has_answer,
            "has_sources": bool(sources),
            "sources_count": _safe_len(sources),
            "has_context": bool(has_context),
            "has_doctor_recommendations": bool(doctor_recs) and _safe_len(doctor_recs) > 0,
            "has_disclaimer": bool(disclaimer),
            "has_run_id": has_run_id,
            "has_trace_id": has_trace_id,
        }
    except Exception as e:
        logger.warning("final_response_runnable_process_inputs failed: %s", type(e).__name__)
        return {"has_answer": False, "has_sources": False}


def final_response_runnable_process_outputs(output: Any) -> Dict[str, Any]:
    """Extract only safe metadata from final_response runnable outputs.

    Args:
        output: RAGResponse object

    Returns:
        Dict with only safe metadata.
    """
    return rag_chain_process_outputs(output)
