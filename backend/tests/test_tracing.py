"""Tests for LangSmith tracing integration."""
import os
import sys
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import uuid


# =============================================================================
# Local Mock LangSmith SDK for Deterministic Testing
# =============================================================================

@dataclass
class MockRun:
    """Mock run representing a LangSmith run tree."""
    name: str
    run_type: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    run_id: str
    trace_id: str
    parent_run_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class LocalTracer:
    """Local in-memory tracer that mimics LangSmith SDK behavior.
    
    This replaces the real LangSmith client for deterministic testing.
    It creates a run tree hierarchy with parent/child relationships
    and applies process_inputs/process_outputs sanitizers.
    """
    
    def __init__(self):
        self.runs: List[MockRun] = []
        self._stack: List[MockRun] = []
        self._run_counter = 0
    
    def _new_id(self) -> str:
        self._run_counter += 1
        return f"run-{self._run_counter:04d}-{uuid.uuid4().hex[:8]}"
    
    def _new_trace_id(self) -> str:
        return f"trace-{uuid.uuid4().hex[:12]}"
    
    def start_run(self, name: str, run_type: str, inputs: Dict[str, Any],
                  process_inputs=None, process_outputs=None,
                  tags=None, metadata=None, parent=None) -> MockRun:
        """Start a new run (mimics @traceable function entry)."""
        # Apply input sanitizer
        sanitized_inputs = process_inputs(inputs) if process_inputs else inputs
        
        trace_id = parent.trace_id if parent else self._new_trace_id()
        parent_run_id = parent.run_id if parent else None
        
        run = MockRun(
            name=name,
            run_type=run_type,
            inputs=sanitized_inputs,
            outputs={},
            run_id=self._new_id(),
            trace_id=trace_id,
            parent_run_id=parent_run_id,
            tags=tags or [],
            metadata=metadata or {},
        )
        run._process_outputs = process_outputs
        self._stack.append(run)
        return run
    
    def end_run(self, run: MockRun, outputs: Any = None, error: Optional[Exception] = None) -> MockRun:
        """End a run (mimics @traceable function exit)."""
        if run._process_outputs and outputs is not None and not error:
            run.outputs = run._process_outputs(outputs)
        else:
            run.outputs = outputs or {}
        if error:
            run.outputs["error"] = str(error)
        self.runs.append(run)
        self._stack.pop()
        return run
    
    def get_current_run(self) -> Optional[MockRun]:
        """Get the currently active run (mimics get_current_run_tree())."""
        return self._stack[-1] if self._stack else None
    
    def clear(self):
        self.runs.clear()
        self._stack.clear()
        self._run_counter = 0
    
    def get_runs_by_name(self, name: str) -> List[MockRun]:
        return [r for r in self.runs if r.name == name]
    
    def get_child_runs(self, parent_run_id: str) -> List[MockRun]:
        return [r for r in self.runs if r.parent_run_id == parent_run_id]
    
    def get_parent_run(self, run: MockRun) -> Optional[MockRun]:
        if not run.parent_run_id:
            return None
        for r in self.runs:
            if r.run_id == run.parent_run_id:
                return r
        return None


# Global local tracer instance
_local_tracer = LocalTracer()


def get_local_tracer() -> LocalTracer:
    return _local_tracer


def create_local_traceable(name: str, run_type: str,
                           process_inputs=None, process_outputs=None,
                           tags=None, metadata=None):
    """Create a traceable decorator that uses the local tracer."""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            inputs = dict(bound_args.arguments)
            inputs.pop('self', None)
            inputs.pop('cls', None)
            
            parent = _local_tracer.get_current_run()
            run = _local_tracer.start_run(
                name=name, run_type=run_type, inputs=inputs,
                process_inputs=process_inputs, process_outputs=process_outputs,
                tags=tags, metadata=metadata, parent=parent
            )
            try:
                result = await func(*args, **kwargs)
                _local_tracer.end_run(run, outputs=result)
                return result
            except Exception as e:
                _local_tracer.end_run(run, error=e)
                raise
        
        def sync_wrapper(*args, **kwargs):
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            inputs = dict(bound_args.arguments)
            inputs.pop('self', None)
            inputs.pop('cls', None)
            
            parent = _local_tracer.get_current_run()
            run = _local_tracer.start_run(
                name=name, run_type=run_type, inputs=inputs,
                process_inputs=process_inputs, process_outputs=process_outputs,
                tags=tags, metadata=metadata, parent=parent
            )
            try:
                result = func(*args, **kwargs)
                _local_tracer.end_run(run, outputs=result)
                return result
            except Exception as e:
                _local_tracer.end_run(run, error=e)
                raise
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# =============================================================================
# Tests for Tracing Disabled
# =============================================================================

class TestTracingDisabled:
    """Tests that tracing being disabled doesn't break functionality."""

    @pytest.fixture(autouse=True)
    def _disable_tracing(self, monkeypatch):
        """Ensure LangSmith tracing is disabled for these tests.
        
        Sets env vars BEFORE importing the chain module so the module
        picks up the disabled tracing configuration.
        """
        monkeypatch.setenv("LANGSMITH_TRACING_V2", "false")
        monkeypatch.setenv("LANGSMITH_TRACING", "false")
        monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
        
        # Reload chain module to pick up disabled tracing
        import importlib
        import app.ai.rag.chain as chain_module
        importlib.reload(chain_module)
        
        yield
        
        # Restore the module to its original state by reloading
        importlib.reload(chain_module)

    @pytest.mark.asyncio
    async def test_rag_chain_generate_works_without_tracing(self):
        """RAGChain.generate() should work normally when tracing is disabled."""
        from app.ai.rag.chain import RAGChain, RAGResponse

        chain = RAGChain()
        chain._retriever = MagicMock()
        chain._retriever.retrieve_with_scores.return_value = []
        chain._llm_provider = MagicMock()
        chain._llm_provider.model = "test-model"
        chain._llm_provider.generate_response = MagicMock()
        chain._llm_provider.generate_response.return_value = MagicMock(
            content="Test answer",
            model="test-model",
            usage={"prompt_tokens": 10, "completion_tokens": 20},
            finish_reason="stop"
        )

        response = await chain.generate(query="What are visiting hours?")

        assert isinstance(response, RAGResponse)
        assert response.answer == "Test answer"
        assert response.model == "test-model"
        assert response.has_context is False
        assert response.langsmith_run_id is None
        assert response.langsmith_trace_id is None

    @pytest.mark.asyncio
    async def test_retriever_works_without_tracing(self):
        from app.ai.rag.retriever import Retriever, SearchResult
        from app.ai.vectorstore.base import VectorDocument

        retriever = Retriever()
        retriever.retrieve = MagicMock(return_value=[
            SearchResult(document=VectorDocument(id="1", content="test", embedding=[0.1]*384, metadata={}), score=0.9)
        ])

        results = retriever.retrieve_with_scores(query="test", top_k=4, min_score=0.5)
        assert len(results) == 1
        assert results[0].score == 0.9

    def test_faiss_search_works_without_tracing(self):
        from app.ai.vectorstore.faiss_store import FAISSVectorStore
        from app.ai.vectorstore.base import SearchResult, VectorDocument

        store = FAISSVectorStore(dimension=384, path="/tmp/test_faiss")
        store._initialized = True
        store._index = MagicMock()
        store._index.ntotal = 1
        store._index.search.return_value = ([[0.9]], [[0]])
        store._id_to_index = {"doc1": 0}
        store._documents = {"doc1": VectorDocument(id="doc1", content="test", embedding=[0.1]*384, metadata={})}

        results = store.search(query_embedding=[0.1]*384, top_k=4)
        assert len(results) == 1
        assert results[0].score == 0.9

    def test_groq_provider_works_without_tracing(self):
        from app.ai.providers.groq_provider import GroqProvider
        from app.ai.providers.base import LLMResponse

        with patch("app.ai.providers.groq_provider.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_response = MagicMock()
            mock_response.choices = [MagicMock(message=MagicMock(content="Test response"), finish_reason="stop")]
            mock_response.model = "llama-3.1-8b-instant"
            mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=20)
            mock_client.chat.completions.create.return_value = mock_response

            provider = GroqProvider(api_key="test-key", model="llama-3.1-8b-instant")
            response = provider.generate_response(messages=[MagicMock(role="user", content="Hello")])

            assert isinstance(response, LLMResponse)
            assert response.content == "Test response"
            assert response.model == "llama-3.1-8b-instant"


# =============================================================================
# Tests for Sanitizer Functions (Unit Tests)
# =============================================================================

class TestSanitizerFunctions:
    """Unit tests for individual sanitizer functions."""

    @pytest.mark.asyncio
    async def test_rag_chain_process_inputs_no_phi(self):
        from app.ai.tracing.config import rag_chain_process_inputs

        inputs = rag_chain_process_inputs({
            "query": "Patient John Doe has fever and chest pain, SSN 123-45-6789",
            "conversation_history": [{"role": "user", "content": "Previous medical question"}],
            "top_k": 4,
            "db": None
        })

        assert "query" not in inputs
        assert "conversation_history" not in inputs
        assert inputs["top_k"] == 4
        assert inputs["has_conversation_history"] is True
        assert inputs["history_length"] == 1
        assert "John Doe" not in str(inputs)
        assert "123-45-6789" not in str(inputs)
        assert "fever" not in str(inputs)
        assert "chest pain" not in str(inputs)

    @pytest.mark.asyncio
    async def test_rag_chain_process_outputs_no_phi(self):
        from app.ai.tracing.config import rag_chain_process_outputs
        from app.ai.rag.chain import RAGResponse

        response = RAGResponse(
            answer="Patient John Doe should take 500mg paracetamol for fever",
            sources=[{"document_id": "doc1", "source": "test.pdf", "page": "1", "score": 0.9, "content_preview": "..."}],
            has_context=True,
            model="llama-3.1-8b-instant",
            doctor_recommendations=[{"doctor_id": "1", "name": "Dr. Smith"}],
            disclaimer="Medical disclaimer"
        )

        outputs = rag_chain_process_outputs(response)
        assert "answer" not in outputs
        assert outputs["has_context"] is True
        assert outputs["sources_count"] == 1
        assert outputs["model"] == "llama-3.1-8b-instant"
        assert outputs["has_doctor_recommendations"] is True
        assert outputs["has_disclaimer"] is True
        assert "score_summary" in outputs
        assert "John Doe" not in str(outputs)
        assert "paracetamol" not in str(outputs)
        assert "fever" not in str(outputs)

    def test_retriever_process_inputs_no_phi(self):
        from app.ai.tracing.config import retriever_process_inputs

        inputs = retriever_process_inputs({
            "query": "Patient Jane Smith has diabetes, MRN 987654",
            "top_k": 4,
            "filter_metadata": {"source": "medical_records"}
        })

        assert "query" not in inputs
        assert inputs["top_k"] == 4
        assert inputs["has_filter"] is True
        assert "Jane Smith" not in str(inputs)
        assert "diabetes" not in str(inputs)
        assert "987654" not in str(inputs)

    def test_faiss_search_process_inputs_no_embedding_content(self):
        from app.ai.tracing.config import faiss_search_process_inputs

        mock_index = MagicMock()
        mock_index.ntotal = 1000

        inputs = faiss_search_process_inputs({
            "query_embedding": [0.1] * 384,
            "top_k": 4,
            "filter_metadata": {"category": "medical"},
            "self": type('obj', (object,), {'_index': mock_index})()
        })

        assert inputs["top_k"] == 4
        assert inputs["embedding_dim"] == 384
        assert inputs["has_filter"] is True
        assert inputs["index_size"] == 1000
        assert "[0.1" not in str(inputs)

    def test_groq_llm_process_inputs_no_prompts(self):
        from app.ai.tracing.config import groq_llm_process_inputs
        from app.ai.providers.base import LLMMessage

        mock_self = type('obj', (object,), {'model': "llama-3.1-8b-instant", 'temperature': 0.2, 'max_tokens': 2048})()
        messages = [
            LLMMessage(role="system", content="You are a medical assistant. Patient John Doe has cancer."),
            LLMMessage(role="user", content="What treatment for patient SSN 111-22-3333?")
        ]

        inputs = groq_llm_process_inputs({
            "messages": messages,
            "system_prompt": "System prompt with patient data",
            "temperature": 0.5,
            "max_tokens": 1000,
            "self": mock_self
        })

        assert inputs["model"] == "llama-3.1-8b-instant"
        assert inputs["temperature"] == 0.5
        assert inputs["max_tokens"] == 1000
        assert inputs["message_count"] == 2
        assert inputs["has_system_prompt"] is True
        assert "John Doe" not in str(inputs)
        assert "cancer" not in str(inputs)
        assert "111-22-3333" not in str(inputs)
        assert "medical assistant" not in str(inputs)

    def test_groq_llm_process_outputs_only_usage(self):
        from app.ai.tracing.config import groq_llm_process_outputs
        from app.ai.providers.base import LLMResponse

        response = LLMResponse(
            content="Patient Jane Smith should undergo chemotherapy for breast cancer",
            model="llama-3.1-8b-instant",
            usage={"prompt_tokens": 100, "completion_tokens": 50},
            finish_reason="stop"
        )

        outputs = groq_llm_process_outputs(response)
        assert outputs["model"] == "llama-3.1-8b-instant"
        assert outputs["finish_reason"] == "stop"
        assert outputs["usage"]["prompt_tokens"] == 100
        assert outputs["usage"]["completion_tokens"] == 50
        assert "Jane Smith" not in str(outputs)
        assert "chemotherapy" not in str(outputs)
        assert "breast cancer" not in str(outputs)

    def test_rag_chain_process_outputs_dict_with_phi(self):
        """Test rag_chain_process_outputs with dict containing PHI - should sanitize."""
        from app.ai.tracing.config import rag_chain_process_outputs

        output_dict = {
            "answer": "Patient John Doe should take 500mg paracetamol for fever",
            "has_context": True,
            "sources": [
                {"document_id": "doc1", "source": "test.pdf", "page": "1", "score": 0.9, "content_preview": "Patient Jane Smith has cancer..."}
            ],
            "model": "llama-3.1-8b-instant",
            "doctor_recommendations": [
                {"doctor_id": "1", "name": "Dr. Smith", "specialization": "Cardiology"}
            ],
            "disclaimer": "Medical disclaimer",
        }

        outputs = rag_chain_process_outputs(output_dict)

        assert outputs["has_context"] is True
        assert outputs["sources_count"] == 1
        assert outputs["model"] == "llama-3.1-8b-instant"
        assert outputs["has_doctor_recommendations"] is True
        assert outputs["has_disclaimer"] is True
        assert "score_summary" in outputs
        assert "John Doe" not in str(outputs)
        assert "Jane Smith" not in str(outputs)
        assert "paracetamol" not in str(outputs)
        assert "fever" not in str(outputs)
        assert "cancer" not in str(outputs)
        assert "Dr. Smith" not in str(outputs)
        assert "Cardiology" not in str(outputs)
        assert "500mg" not in str(outputs)

    def test_rag_chain_process_outputs_dict_missing_optional_fields(self):
        """Test rag_chain_process_outputs with dict missing optional fields."""
        from app.ai.tracing.config import rag_chain_process_outputs

        output_dict = {
            "has_context": False,
            "model": "test-model",
        }

        outputs = rag_chain_process_outputs(output_dict)

        assert outputs["has_context"] is False
        assert outputs["sources_count"] == 0
        assert outputs["model"] == "test-model"
        assert outputs["has_doctor_recommendations"] is False
        assert outputs["has_disclaimer"] is False
        assert "score_summary" in outputs

    def test_rag_chain_process_outputs_ragresponse_object(self):
        """Test rag_chain_process_outputs with RAGResponse object (existing behavior)."""
        from app.ai.tracing.config import rag_chain_process_outputs
        from app.ai.rag.chain import RAGResponse

        response = RAGResponse(
            answer="Patient John Doe should take 500mg paracetamol for fever",
            sources=[
                {"document_id": "doc1", "source": "test.pdf", "page": "1", "score": 0.9, "content_preview": "..."}
            ],
            has_context=True,
            model="llama-3.1-8b-instant",
            doctor_recommendations=[{"doctor_id": "1", "name": "Dr. Smith"}],
            disclaimer="Medical disclaimer"
        )

        outputs = rag_chain_process_outputs(response)

        assert outputs["has_context"] is True
        assert outputs["sources_count"] == 1
        assert outputs["model"] == "llama-3.1-8b-instant"
        assert outputs["has_doctor_recommendations"] is True
        assert outputs["has_disclaimer"] is True
        assert "score_summary" in outputs
        assert "John Doe" not in str(outputs)
        assert "paracetamol" not in str(outputs)
        assert "fever" not in str(outputs)

    def test_rag_chain_process_outputs_fail_closed_on_bad_dict(self):
        """Test rag_chain_process_outputs fail-closed behavior with problematic dict."""
        from app.ai.tracing.config import rag_chain_process_outputs

        class BadDict(dict):
            def get(self, key, default=None):
                if key == "has_context":
                    raise RuntimeError("Cannot access has_context")
                if key == "sources":
                    raise RuntimeError("Cannot access sources")
                return default

        outputs = rag_chain_process_outputs(BadDict())
        assert isinstance(outputs, dict)
        assert outputs["has_context"] is False
        assert outputs["sources_count"] == 0
        assert outputs["model"] == ""
        assert outputs["has_doctor_recommendations"] is False
        assert outputs["has_disclaimer"] is False


# =============================================================================
# Tests for LCEL Runnable Sanitizers (Unit Tests)
# =============================================================================

class TestLCELRunnableSanitizers:
    """Unit tests for LCEL runnable sanitizer functions."""

    def test_retriever_runnable_process_inputs(self):
        from app.ai.tracing.config import retriever_runnable_process_inputs

        inputs = retriever_runnable_process_inputs({
            "query": "Patient has cancer", "top_k": 4, "conversation_history": [{"role": "user", "content": "Previous"}],
        })
        assert "query" not in inputs
        assert "conversation_history" not in inputs
        assert inputs["top_k"] == 4

    def test_retriever_runnable_process_outputs(self):
        from app.ai.tracing.config import retriever_runnable_process_outputs
        from app.ai.vectorstore.base import SearchResult, VectorDocument

        results = [
            SearchResult(document=VectorDocument(id="1", content="Patient John Doe has cancer", embedding=[0.1]*384, metadata={}), score=0.9),
            SearchResult(document=VectorDocument(id="2", content="Treatment for cancer", embedding=[0.2]*384, metadata={}), score=0.8),
        ]
        outputs = retriever_runnable_process_outputs({"results": results})
        assert outputs["results_count"] == 2
        assert outputs["score_summary"]["min"] == 0.8
        assert outputs["score_summary"]["max"] == 0.9
        assert "John Doe" not in str(outputs)
        assert "cancer" not in str(outputs)
        assert "Treatment" not in str(outputs)

    def test_is_medical_runnable_process_inputs(self):
        from app.ai.tracing.config import is_medical_runnable_process_inputs
        inputs = is_medical_runnable_process_inputs({
            "query": "Patient has cancer", "results": [{"score": 0.9}], "conversation_history": [{"role": "user", "content": "Previous"}],
        })
        assert "query" not in inputs
        assert "results" not in inputs
        assert "conversation_history" not in inputs
        assert inputs["has_results"] is True

    def test_is_medical_runnable_process_outputs(self):
        from app.ai.tracing.config import is_medical_runnable_process_outputs
        outputs = is_medical_runnable_process_outputs({"is_medical": True})
        assert outputs["is_medical"] is True

    def test_detect_specialization_runnable_process_inputs(self):
        from app.ai.tracing.config import detect_specialization_runnable_process_inputs
        inputs = detect_specialization_runnable_process_inputs({
            "query": "Patient has heart pain", "is_medical": True, "results": [],
        })
        assert "query" not in inputs
        assert "results" not in inputs
        assert inputs["is_medical"] is True

    def test_detect_specialization_runnable_process_outputs(self):
        from app.ai.tracing.config import detect_specialization_runnable_process_outputs
        outputs = detect_specialization_runnable_process_outputs({"specializations": ["cardiology", "medicine"]})
        assert outputs["specializations_count"] == 2

    def test_build_prompt_runnable_process_inputs(self):
        from app.ai.tracing.config import build_prompt_runnable_process_inputs
        inputs = build_prompt_runnable_process_inputs({
            "query": "Patient John Doe has chest pain", "results": [{"score": 0.9}, {"score": 0.8}],
            "conversation_history": [{"role": "user", "content": "Previous"}], "specializations": ["cardiology"], "is_medical": True,
        })
        assert "query" not in inputs
        assert "results" not in inputs
        assert "conversation_history" not in inputs
        assert "specializations" not in inputs
        assert "is_medical" not in inputs
        assert inputs["has_results"] is True
        assert inputs["results_count"] == 2
        assert inputs["has_history"] is True
        assert inputs["history_length"] == 1

    def test_build_prompt_runnable_process_outputs(self):
        from app.ai.tracing.config import build_prompt_runnable_process_outputs
        outputs = build_prompt_runnable_process_outputs({
            "system_prompt": "You are a doctor. Patient John Doe has cancer.",
            "user_prompt": "What treatment for SSN 111-22-3333?",
            "has_context": True,
        })
        assert outputs["has_context"] is True
        assert outputs["has_system_prompt"] is True
        assert outputs["has_user_prompt"] is True
        assert "John Doe" not in str(outputs)
        assert "cancer" not in str(outputs)
        assert "111-22-3333" not in str(outputs)

    def test_llm_runnable_process_inputs(self):
        from app.ai.tracing.config import llm_runnable_process_inputs
        inputs = llm_runnable_process_inputs({
            "system_prompt": "You are a doctor. Patient John Doe has cancer.",
            "user_prompt": "What treatment for SSN 111-22-3333?",
            "query": "original query",
        })
        assert "system_prompt" not in inputs
        assert "user_prompt" not in inputs
        assert "query" not in inputs
        assert inputs["has_system_prompt"] is True
        assert inputs["has_user_prompt"] is True

    def test_llm_runnable_process_outputs(self):
        from app.ai.tracing.config import llm_runnable_process_outputs
        outputs = llm_runnable_process_outputs({
            "answer": "Patient John Doe should take medication for cancer", "query": "original query",
        })
        assert outputs["has_answer"] is True
        assert "John Doe" not in str(outputs)
        assert "cancer" not in str(outputs)
        assert "medication" not in str(outputs)

    def test_format_sources_runnable_process_inputs(self):
        from app.ai.tracing.config import format_sources_runnable_process_inputs
        inputs = format_sources_runnable_process_inputs({
            "results": [{"score": 0.9}, {"score": 0.8}], "query": "Patient has cancer",
        })
        assert "results" not in inputs
        assert "query" not in inputs
        assert inputs["has_results"] is True
        assert inputs["results_count"] == 2

    def test_format_sources_runnable_process_outputs(self):
        from app.ai.tracing.config import format_sources_runnable_process_outputs
        sources = [
            {"document_id": "1", "source": "test.pdf", "page": "1", "score": 0.9, "content_preview": "Patient John Doe..."},
            {"document_id": "2", "source": "test2.pdf", "page": "2", "score": 0.8, "content_preview": "Cancer treatment..."},
        ]
        outputs = format_sources_runnable_process_outputs(sources)
        assert outputs["sources_count"] == 2
        assert "John Doe" not in str(outputs)
        assert "Cancer" not in str(outputs)

    def test_doctor_recommendations_runnable_process_inputs(self):
        from app.ai.tracing.config import doctor_recommendations_runnable_process_inputs
        inputs = doctor_recommendations_runnable_process_inputs({
            "is_medical": True, "specializations": ["cardiology", "medicine"], "db": None,
            "has_context": True, "query": "Patient has heart pain",
        })
        assert "query" not in inputs
        assert "specializations" not in inputs
        assert "db" not in inputs
        assert inputs["is_medical"] is True
        assert inputs["has_specializations"] is True
        assert inputs["specializations_count"] == 2
        assert inputs["has_context"] is True

    def test_doctor_recommendations_runnable_process_outputs(self):
        from app.ai.tracing.config import doctor_recommendations_runnable_process_outputs
        outputs = doctor_recommendations_runnable_process_outputs({
            "doctor_recommendations": [
                {"doctor_id": "1", "name": "Dr. Smith", "specialization": "Cardiology"},
                {"doctor_id": "2", "name": "Dr. Jones", "specialization": "Medicine"},
            ], "disclaimer": "Medical disclaimer",
        })
        assert outputs["has_doctor_recommendations"] is True
        assert outputs["doctor_recommendations_count"] == 2
        assert outputs["has_disclaimer"] is True
        assert "Dr. Smith" not in str(outputs)
        assert "Cardiology" not in str(outputs)

    def test_final_response_runnable_process_inputs(self):
        from app.ai.tracing.config import final_response_runnable_process_inputs
        inputs = final_response_runnable_process_inputs({
            "answer": "Patient John Doe should take medication", "sources": [{"score": 0.9}],
            "has_context": True, "doctor_recommendations": [{"doctor_id": "1"}], "disclaimer": "Medical disclaimer",
            "langsmith_run_id": "run-123", "langsmith_trace_id": "trace-456", "query": "original query",
        })
        assert "answer" not in inputs
        assert "sources" not in inputs
        assert "doctor_recommendations" not in inputs
        assert "disclaimer" not in inputs
        assert "query" not in inputs
        assert inputs["has_answer"] is True
        assert inputs["has_sources"] is True
        assert inputs["sources_count"] == 1
        assert inputs["has_context"] is True
        assert inputs["has_doctor_recommendations"] is True
        assert inputs["has_disclaimer"] is True
        assert inputs["has_run_id"] is True
        assert inputs["has_trace_id"] is True

    def test_final_response_runnable_process_outputs(self):
        from app.ai.tracing.config import final_response_runnable_process_outputs
        from app.ai.rag.chain import RAGResponse
        response = RAGResponse(
            answer="Patient John Doe should take medication", sources=[{"score": 0.9}],
            has_context=True, model="llama-3.1-8b-instant",
            doctor_recommendations=[{"doctor_id": "1"}], disclaimer="Medical disclaimer",
        )
        outputs = final_response_runnable_process_outputs(response)
        assert outputs["has_context"] is True
        assert outputs["sources_count"] == 1
        assert outputs["model"] == "llama-3.1-8b-instant"
        assert outputs["has_doctor_recommendations"] is True
        assert outputs["has_disclaimer"] is True
        assert "John Doe" not in str(outputs)


# =============================================================================
# Tests for Fail-Closed Behavior
# =============================================================================

class TestSanitizationFailClosed:
    """Tests that verify fail-closed behavior - raw data NEVER leaks on sanitization errors."""

    @pytest.mark.asyncio
    async def test_rag_chain_process_inputs_fail_closed(self):
        from app.ai.tracing.config import rag_chain_process_inputs
        for bad_input in ["not a dict", None, [], 123, object()]:
            result = rag_chain_process_inputs(bad_input)
            assert isinstance(result, dict)
            assert "top_k" in result

        class Unserializable:
            def __repr__(self):
                raise RuntimeError("Cannot serialize")
        result = rag_chain_process_inputs({"query": "test", "top_k": Unserializable()})
        assert isinstance(result, dict)
        assert result["top_k"] is None

    @pytest.mark.asyncio
    async def test_rag_chain_process_outputs_fail_closed(self):
        from app.ai.tracing.config import rag_chain_process_outputs
        for bad_input in ["not an object", None, 123, object()]:
            result = rag_chain_process_outputs(bad_input)
            assert isinstance(result, dict)
            assert result["has_context"] is False
            assert result["sources_count"] == 0

        class BadObject:
            @property
            def has_context(self):
                raise RuntimeError("Fail")
        result = rag_chain_process_outputs(BadObject())
        assert isinstance(result, dict)
        assert result["has_context"] is False

    def test_all_lcel_sanitizers_fail_closed(self):
        from app.ai.tracing.config import (
            retriever_runnable_process_inputs, retriever_runnable_process_outputs,
            is_medical_runnable_process_inputs, is_medical_runnable_process_outputs,
            detect_specialization_runnable_process_inputs, detect_specialization_runnable_process_outputs,
            build_prompt_runnable_process_inputs, build_prompt_runnable_process_outputs,
            llm_runnable_process_inputs, llm_runnable_process_outputs,
            format_sources_runnable_process_inputs, format_sources_runnable_process_outputs,
            doctor_recommendations_runnable_process_inputs, doctor_recommendations_runnable_process_outputs,
            final_response_runnable_process_inputs, final_response_runnable_process_outputs,
        )
        input_sanitizers = [
            retriever_runnable_process_inputs, is_medical_runnable_process_inputs,
            detect_specialization_runnable_process_inputs, build_prompt_runnable_process_inputs,
            llm_runnable_process_inputs, format_sources_runnable_process_inputs,
            doctor_recommendations_runnable_process_inputs, final_response_runnable_process_inputs,
        ]
        output_sanitizers = [
            retriever_runnable_process_outputs, is_medical_runnable_process_outputs,
            detect_specialization_runnable_process_outputs, build_prompt_runnable_process_outputs,
            llm_runnable_process_outputs, format_sources_runnable_process_outputs,
            doctor_recommendations_runnable_process_outputs, final_response_runnable_process_outputs,
        ]
        for bad_input in ["string", None, [], 123, object()]:
            for sanitizer in input_sanitizers:
                result = sanitizer(bad_input)
                assert isinstance(result, dict), f"{sanitizer.__name__} must return dict"
            for sanitizer in output_sanitizers:
                result = sanitizer(bad_input)
                assert isinstance(result, dict), f"{sanitizer.__name__} must return dict"


# =============================================================================
# Integration Tests: Actual LCEL Chain with Local Tracer
# =============================================================================

class TestNestedLCELChainTracing:
    """Integration tests that execute the real LCEL chain with the local tracer
    and verify actual trace payloads are sanitized at every level."""

    @pytest.fixture(autouse=True)
    def setup_local_tracer(self):
        """Set up local tracer by patching langsmith.traceable before importing modules."""
        _local_tracer.clear()
        
        # Patch langsmith.traceable and get_current_run_tree at the source
        import langsmith
        original_traceable = langsmith.traceable
        original_get_current_run_tree = langsmith.get_current_run_tree
        
        def mock_traceable(name=None, run_type=None, process_inputs=None, process_outputs=None, **kwargs):
            def decorator(func):
                return create_local_traceable(name, run_type, process_inputs, process_outputs, **kwargs)(func)
            return decorator
        
        langsmith.traceable = mock_traceable
        
        # Now import modules - they'll get the patched traceable
        import importlib
        import app.ai.rag.chain as chain_module
        import app.ai.rag.retriever as retriever_module
        import app.ai.vectorstore.faiss_store as faiss_module
        import app.ai.providers.groq_provider as groq_module
        
        # Store original modules for restoration
        original_chain = sys.modules.get("app.ai.rag.chain")
        original_retriever = sys.modules.get("app.ai.rag.retriever")
        original_faiss = sys.modules.get("app.ai.vectorstore.faiss_store")
        original_groq = sys.modules.get("app.ai.providers.groq_provider")
        
        # Reload to pick up patched traceable
        importlib.reload(chain_module)
        importlib.reload(retriever_module)
        importlib.reload(faiss_module)
        importlib.reload(groq_module)
        
        # Also patch get_current_run_tree to use local tracer
        original_get_current_run_tree_chain = chain_module.get_current_run_tree
        
        def mock_get_current_run_tree():
            run = _local_tracer.get_current_run()
            if run:
                class MockRunTree:
                    def __init__(self, run):
                        self.id = run.run_id
                        self.trace_id = run.trace_id
                return MockRunTree(run)
            return None
        
        chain_module.get_current_run_tree = mock_get_current_run_tree
        
        yield
        
        # Restore all patched state
        langsmith.traceable = original_traceable
        langsmith.get_current_run_tree = original_get_current_run_tree
        chain_module.get_current_run_tree = original_get_current_run_tree_chain
        
        # Restore original modules
        if original_chain:
            sys.modules["app.ai.rag.chain"] = original_chain
        if original_retriever:
            sys.modules["app.ai.rag.retriever"] = original_retriever
        if original_faiss:
            sys.modules["app.ai.vectorstore.faiss_store"] = original_faiss
        if original_groq:
            sys.modules["app.ai.providers.groq_provider"] = original_groq
        
        _local_tracer.clear()

    @pytest.mark.asyncio
    async def test_full_chain_creates_trace_hierarchy(self):
        """Execute the real LCEL chain and verify parent/child run hierarchy."""
        from app.ai.rag.chain import RAGChain
        from app.ai.providers.base import LLMResponse
        from app.ai.vectorstore.base import SearchResult, VectorDocument

        chain = RAGChain()
        mock_retriever = MagicMock()
        mock_search_result = MagicMock(spec=SearchResult)
        mock_search_result.document = MagicMock(spec=VectorDocument)
        mock_search_result.document.id = "doc1"
        mock_search_result.document.content = "Hospital visiting hours are 9 AM to 5 PM daily."
        mock_search_result.document.metadata = {"source": "hospital_info.pdf", "page": 1}
        mock_search_result.score = 0.85
        mock_retriever.retrieve_with_scores.return_value = [mock_search_result]
        chain._retriever = mock_retriever

        chain._llm_provider = MagicMock()
        chain._llm_provider.model = "llama-3.1-8b-instant"
        chain._llm_provider.generate_response = MagicMock(return_value=LLMResponse(
            content="Visiting hours are 9 AM to 5 PM daily.",
            model="llama-3.1-8b-instant",
            usage={"prompt_tokens": 50, "completion_tokens": 30},
            finish_reason="stop"
        ))

        response = await chain.generate(query="What are visiting hours?", top_k=4)

        assert response.answer == "Visiting hours are 9 AM to 5 PM daily."
        assert response.has_context is True
        assert len(response.sources) == 1

        # Verify trace hierarchy
        tracer = get_local_tracer()
        parent_runs = tracer.get_runs_by_name("rag_generate")
        assert len(parent_runs) == 1, "Exactly one parent rag_generate run"
        parent = parent_runs[0]
        assert parent.run_type == "chain"

        # Verify child runs exist
        child_names = {run.name for run in tracer.get_child_runs(parent.run_id)}
        expected_children = {
            "retriever_runnable", "is_medical_runnable", "detect_specialization_runnable",
            "build_prompt_runnable", "llm_runnable", "format_sources_runnable",
            "doctor_recommendations_runnable", "final_response_runnable"
        }
        for expected in expected_children:
            assert expected in child_names, f"Missing child run: {expected}"

    @pytest.mark.asyncio
    async def test_no_phi_in_any_trace_payload(self):
        """Verify NO PHI appears in ANY run's inputs or outputs in the trace hierarchy.
        
        This is the critical privacy test - it executes the actual chain with
        PHI at every level and verifies the local tracer captures sanitized payloads.
        """
        from app.ai.rag.chain import RAGChain
        from app.ai.providers.base import LLMResponse
        from app.ai.vectorstore.base import SearchResult, VectorDocument

        chain = RAGChain()

        # Mock retriever with PHI in retrieved content
        mock_retriever = MagicMock()
        mock_search_result = MagicMock(spec=SearchResult)
        mock_search_result.document = MagicMock(spec=VectorDocument)
        mock_search_result.document.id = "doc1"
        mock_search_result.document.content = "Patient John Doe (SSN: 123-45-6789) has cancer and needs chemotherapy."
        mock_search_result.document.metadata = {"source": "medical_records.pdf", "page": 1, "patient_id": "PAT-123"}
        mock_search_result.score = 0.9
        mock_retriever.retrieve_with_scores.return_value = [mock_search_result]
        chain._retriever = mock_retriever

        # Mock LLM provider with PHI in response
        chain._llm_provider = MagicMock()
        chain._llm_provider.model = "llama-3.1-8b-instant"
        chain._llm_provider.generate_response = MagicMock(return_value=LLMResponse(
            content="Patient John Doe should undergo chemotherapy for cancer treatment. SSN 123-45-6789 on file.",
            model="llama-3.1-8b-instant",
            usage={"prompt_tokens": 100, "completion_tokens": 80},
            finish_reason="stop"
        ))

        # Execute with PHI in query and conversation history
        response = await chain.generate(
            query="Patient John Doe has cancer, SSN 123-45-6789, what treatment?",
            conversation_history=[
                {"role": "user", "content": "Previous: patient Jane Smith MRN 987654 has diabetes"},
                {"role": "assistant", "content": "Patient Jane Smith should take metformin"}
            ],
            top_k=4
        )

        assert "chemotherapy" in response.answer
        assert response.has_context is True

        # PHI strings that must NOT appear in ANY captured run
        phi_strings = [
            "John Doe", "Jane Smith", "123-45-6789", "987654",
            "cancer", "diabetes", "chemotherapy", "metformin",
            "SSN", "MRN", "patient", "PAT-123", "treatment",
        ]

        tracer = get_local_tracer()
        for run in tracer.runs:
            run_str = str(run.inputs) + str(run.outputs)
            for phi in phi_strings:
                assert phi not in run_str, (
                    f"PHI '{phi}' LEAKED in run '{run.name}' (type: {run.run_type}). "
                    f"inputs={run.inputs}, outputs={run.outputs}"
                )

    @pytest.mark.asyncio
    async def test_regression_comprehensive_phi_absent(self):
        """Regression test with comprehensive sensitive data at every level."""
        from app.ai.rag.chain import RAGChain
        from app.ai.providers.base import LLMResponse
        from app.ai.vectorstore.base import SearchResult, VectorDocument

        chain = RAGChain()

        mock_retriever = MagicMock()
        mock_search_result = MagicMock(spec=SearchResult)
        mock_search_result.document = MagicMock(spec=VectorDocument)
        mock_search_result.document.id = "doc1"
        mock_search_result.document.content = (
            "Patient Alice Johnson, DOB 1980-05-15, MRN: MED-789456, "
            "diagnosed with breast cancer (ICD-10: C50.911), "
            "prescribed tamoxifen 20mg daily. "
            "Contact: Dr. Robert Wilson, MD, Oncology Dept, Phone: 555-0199"
        )
        mock_search_result.document.metadata = {
            "source": "oncology_records.pdf", "page": 5, "patient_id": "PAT-789456", "confidential": True
        }
        mock_search_result.score = 0.95
        mock_retriever.retrieve_with_scores.return_value = [mock_search_result]
        chain._retriever = mock_retriever

        chain._llm_provider = MagicMock()
        chain._llm_provider.model = "llama-3.1-8b-instant"
        chain._llm_provider.generate_response = MagicMock(return_value=LLMResponse(
            content=(
                "Alice Johnson should continue tamoxifen 20mg for breast cancer. "
                "Follow up with Dr. Wilson at 555-0199. MRN MED-789456 referenced."
            ),
            model="llama-3.1-8b-instant",
            usage={"prompt_tokens": 150, "completion_tokens": 100},
            finish_reason="stop"
        ))

        await chain.generate(
            query="Alice Johnson DOB 1980-05-15 MRN MED-789456 breast cancer tamoxifen dosage",
            conversation_history=[
                {"role": "user", "content": "Patient Alice Johnson, SSN 999-88-7777, previous visit"},
                {"role": "assistant", "content": "Alice Johnson was prescribed tamoxifen for breast cancer"}
            ],
            top_k=4
        )

        sensitive_values = [
            "Alice Johnson", "Johnson", "Alice", "1980-05-15", "DOB",
            "MED-789456", "MRN", "PAT-789456", "999-88-7777", "SSN",
            "breast cancer", "cancer", "tamoxifen", "20mg", "ICD-10", "C50.911",
            "Dr. Robert Wilson", "Robert Wilson", "Dr. Wilson", "Wilson",
            "Oncology Dept", "555-0199", "confidential", "prescribed", "diagnosed", "follow up",
        ]

        tracer = get_local_tracer()
        for run in tracer.runs:
            run_str = str(run.inputs) + str(run.outputs)
            for sensitive in sensitive_values:
                assert sensitive not in run_str, (
                    f"REGRESSION FAILURE: '{sensitive}' leaked in run '{run.name}' "
                    f"(type: {run.run_type}). inputs={run.inputs}, outputs={run.outputs}"
                )

    @pytest.mark.asyncio
    async def test_sanitizer_outputs_match_expected_schema(self):
        """Verify each run's sanitized outputs match expected schema."""
        from app.ai.rag.chain import RAGChain
        from app.ai.providers.base import LLMResponse
        from app.ai.vectorstore.base import SearchResult, VectorDocument

        chain = RAGChain()
        mock_retriever = MagicMock()
        mock_search_result = MagicMock(spec=SearchResult)
        mock_search_result.document = MagicMock(spec=VectorDocument)
        mock_search_result.document.id = "doc1"
        mock_search_result.document.content = "General hospital info."
        mock_search_result.document.metadata = {"source": "info.pdf", "page": 1}
        mock_search_result.score = 0.8
        mock_retriever.retrieve_with_scores.return_value = [mock_search_result]
        chain._retriever = mock_retriever

        chain._llm_provider = MagicMock()
        chain._llm_provider.model = "llama-3.1-8b-instant"
        chain._llm_provider.generate_response = MagicMock(return_value=LLMResponse(
            content="General answer.", model="llama-3.1-8b-instant",
            usage={"prompt_tokens": 20, "completion_tokens": 15}, finish_reason="stop"
        ))

        await chain.generate(query="What are the visiting hours?", top_k=4)

        tracer = get_local_tracer()
        
        # Verify each run has expected sanitized fields
        run_map = {r.name: r for r in tracer.runs}
        
        # retriever_runnable: inputs has top_k, outputs has results_count + score_summary
        assert "top_k" in run_map["retriever_runnable"].inputs
        assert "query" not in run_map["retriever_runnable"].inputs
        assert "results_count" in run_map["retriever_runnable"].outputs
        assert "score_summary" in run_map["retriever_runnable"].outputs
        
        # is_medical_runnable: outputs has is_medical
        assert "is_medical" in run_map["is_medical_runnable"].outputs
        
        # detect_specialization_runnable: outputs has specializations_count
        assert "specializations_count" in run_map["detect_specialization_runnable"].outputs
        
        # build_prompt_runnable: outputs has has_context, has_system_prompt, has_user_prompt
        assert "has_context" in run_map["build_prompt_runnable"].outputs
        assert "has_system_prompt" in run_map["build_prompt_runnable"].outputs
        assert "has_user_prompt" in run_map["build_prompt_runnable"].outputs
        
        # llm_runnable: outputs has has_answer
        assert "has_answer" in run_map["llm_runnable"].outputs
        
        # format_sources_runnable: outputs has sources_count
        assert "sources_count" in run_map["format_sources_runnable"].outputs
        
        # doctor_recommendations_runnable: outputs has has_doctor_recommendations, has_disclaimer
        assert "has_doctor_recommendations" in run_map["doctor_recommendations_runnable"].outputs
        assert "has_disclaimer" in run_map["doctor_recommendations_runnable"].outputs
        
        # final_response_runnable: outputs has rag_chain_process_outputs fields
        final_out = run_map["final_response_runnable"].outputs
        assert "has_context" in final_out
        assert "sources_count" in final_out
        assert "model" in final_out


# =============================================================================
# Tests for Trace ID Correctness (with local tracer)
# =============================================================================

class TestTraceIdCorrectness:
    """Tests that verify langsmith_run_id and langsmith_trace_id refer to the 
    parent RAGChain.generate() execution, not child LCEL runnable runs.
    
    These tests use the local tracer to verify trace ID behavior.
    """

    @pytest.fixture(autouse=True)
    def setup_local_tracer_for_trace_ids(self):
        """Set up local tracer by patching langsmith.traceable and langsmith.get_current_run_tree before importing."""
        _local_tracer.clear()
        
        import langsmith
        original_traceable = langsmith.traceable
        original_get_current_run_tree = langsmith.get_current_run_tree
        
        def mock_traceable(name=None, run_type=None, process_inputs=None, process_outputs=None, **kwargs):
            def decorator(func):
                return create_local_traceable(name, run_type, process_inputs, process_outputs, **kwargs)(func)
            return decorator
        
        def mock_get_current_run_tree():
            run = _local_tracer.get_current_run()
            if run:
                class MockRunTree:
                    def __init__(self, run):
                        self.id = run.run_id
                        self.trace_id = run.trace_id
                return MockRunTree(run)
            return None
        
        langsmith.traceable = mock_traceable
        langsmith.get_current_run_tree = mock_get_current_run_tree
        
        import importlib
        import app.ai.rag.chain as chain_module
        import app.ai.rag.retriever as retriever_module
        import app.ai.vectorstore.faiss_store as faiss_module
        import app.ai.providers.groq_provider as groq_module
        
        # Store original modules for restoration
        original_chain = sys.modules.get("app.ai.rag.chain")
        original_retriever = sys.modules.get("app.ai.rag.retriever")
        original_faiss = sys.modules.get("app.ai.vectorstore.faiss_store")
        original_groq = sys.modules.get("app.ai.providers.groq_provider")
        
        importlib.reload(chain_module)
        importlib.reload(retriever_module)
        importlib.reload(faiss_module)
        importlib.reload(groq_module)
        
        # Also patch the imported get_current_run_tree in chain_module directly
        # since it imports the function at module level
        original_get_current_run_tree_chain = chain_module.get_current_run_tree
        chain_module.get_current_run_tree = mock_get_current_run_tree
        
        yield
        
        # Restore all patched state
        langsmith.traceable = original_traceable
        langsmith.get_current_run_tree = original_get_current_run_tree
        chain_module.get_current_run_tree = original_get_current_run_tree_chain
        
        # Restore original modules
        if original_chain:
            sys.modules["app.ai.rag.chain"] = original_chain
        if original_retriever:
            sys.modules["app.ai.rag.retriever"] = original_retriever
        if original_faiss:
            sys.modules["app.ai.vectorstore.faiss_store"] = original_faiss
        if original_groq:
            sys.modules["app.ai.providers.groq_provider"] = original_groq
        
        _local_tracer.clear()

    @pytest.mark.asyncio
    async def test_trace_ids_match_parent_generate_execution(self):
        """Verify returned trace IDs correspond to the outer generate() call.
        
        The @traceable decorator on generate() creates the parent run tree.
        Inside generate(), get_current_run_tree() returns this parent run.
        The captured IDs should match what's returned in the response.
        """
        from app.ai.rag.chain import RAGChain
        from app.ai.providers.base import LLMResponse
        from app.ai.vectorstore.base import SearchResult, VectorDocument

        chain = RAGChain()
        mock_retriever = MagicMock()
        mock_search_result = MagicMock(spec=SearchResult)
        mock_search_result.document = MagicMock(spec=VectorDocument)
        mock_search_result.document.id = "doc1"
        mock_search_result.document.content = "Test document content"
        mock_search_result.document.metadata = {"source": "test.pdf", "page": 1}
        mock_search_result.score = 0.85
        mock_retriever.retrieve_with_scores.return_value = [mock_search_result]
        chain._retriever = mock_retriever

        chain._llm_provider = MagicMock()
        chain._llm_provider.model = "llama-3.1-8b-instant"
        chain._llm_provider.generate_response = MagicMock(return_value=LLMResponse(
            content="Test answer", model="llama-3.1-8b-instant",
            usage={"prompt_tokens": 50, "completion_tokens": 30}, finish_reason="stop"
        ))

        response = await chain.generate(query="What are visiting hours?", top_k=4)

        assert response.answer == "Test answer"
        assert response.has_context is True
        assert len(response.sources) == 1

        # Verify trace IDs match parent run
        parent_runs = _local_tracer.get_runs_by_name("rag_generate")
        assert len(parent_runs) == 1
        parent = parent_runs[0]
        
        assert response.langsmith_run_id == parent.run_id, (
            f"langsmith_run_id ({response.langsmith_run_id}) should match "
            f"parent run_id ({parent.run_id})"
        )
        assert response.langsmith_trace_id == parent.trace_id, (
            f"langsmith_trace_id ({response.langsmith_trace_id}) should match "
            f"parent trace_id ({parent.trace_id})"
        )
        assert response.langsmith_run_id is not None
        assert response.langsmith_trace_id is not None

    @pytest.mark.asyncio
    async def test_returned_ids_are_parent_not_child(self):
        """Verify returned IDs are from parent generate(), not from any child runnable."""
        from app.ai.rag.chain import RAGChain
        from app.ai.providers.base import LLMResponse
        from app.ai.vectorstore.base import SearchResult, VectorDocument

        chain = RAGChain()
        mock_retriever = MagicMock()
        mock_search_result = MagicMock(spec=SearchResult)
        mock_search_result.document = MagicMock(spec=VectorDocument)
        mock_search_result.document.id = "doc1"
        mock_search_result.document.content = "Test content"
        mock_search_result.document.metadata = {"source": "test.pdf", "page": 1}
        mock_search_result.score = 0.85
        mock_retriever.retrieve_with_scores.return_value = [mock_search_result]
        chain._retriever = mock_retriever

        chain._llm_provider = MagicMock()
        chain._llm_provider.model = "llama-3.1-8b-instant"
        chain._llm_provider.generate_response = MagicMock(return_value=LLMResponse(
            content="Test answer", model="llama-3.1-8b-instant",
            usage={"prompt_tokens": 50, "completion_tokens": 30}, finish_reason="stop"
        ))

        response = await chain.generate(query="Test query", top_k=4)

        # Get all run IDs in the trace
        all_run_ids = [r.run_id for r in _local_tracer.runs]
        parent_run = _local_tracer.get_runs_by_name("rag_generate")[0]
        
        # Response IDs must match parent, not any child
        assert response.langsmith_run_id == parent_run.run_id, (
            f"Response run_id must be parent ({parent_run.run_id}), not child. "
            f"Got: {response.langsmith_run_id}, all runs: {all_run_ids}"
        )
        assert response.langsmith_trace_id == parent_run.trace_id
        
        # Verify children have DIFFERENT run_ids but SAME trace_id
        children = _local_tracer.get_child_runs(parent_run.run_id)
        for child in children:
            assert child.run_id != parent_run.run_id, f"Child {child.name} has same run_id as parent"
            assert child.trace_id == parent_run.trace_id, f"Child {child.name} has different trace_id"


# =============================================================================
# Tests for Trace ID Correctness (without local tracer - real behavior)
# =============================================================================

class TestTraceIdCorrectnessReal:
    """Tests for trace ID behavior with real LangSmith SDK (no local tracer)."""

    @pytest.mark.asyncio
    async def test_trace_ids_none_when_tracing_disabled(self, monkeypatch):
        """Verify trace IDs are None when LangSmith tracing is disabled via env vars.
        
        Sets env vars BEFORE importing the chain module so the module
        picks up the disabled tracing configuration.
        """
        # Use monkeypatch to set environment variables BEFORE import/reload
        monkeypatch.setenv("LANGSMITH_TRACING_V2", "false")
        monkeypatch.setenv("LANGSMITH_TRACING", "false")
        monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
        
        # Reload modules to ensure clean state with disabled tracing
        import importlib
        import app.ai.rag.chain as chain_module
        importlib.reload(chain_module)
        
        from app.ai.rag.chain import RAGChain
        from app.ai.providers.base import LLMResponse
        from app.ai.vectorstore.base import SearchResult, VectorDocument

        chain = RAGChain()
        mock_retriever = MagicMock()
        mock_search_result = MagicMock(spec=SearchResult)
        mock_search_result.document = MagicMock(spec=VectorDocument)
        mock_search_result.document.id = "doc1"
        mock_search_result.document.content = "Test document content"
        mock_search_result.document.metadata = {"source": "test.pdf", "page": 1}
        mock_search_result.score = 0.85
        mock_retriever.retrieve_with_scores.return_value = [mock_search_result]
        chain._retriever = mock_retriever

        chain._llm_provider = MagicMock()
        chain._llm_provider.model = "llama-3.1-8b-instant"
        chain._llm_provider.generate_response = MagicMock(return_value=LLMResponse(
            content="Test answer", model="llama-3.1-8b-instant",
            usage={"prompt_tokens": 50, "completion_tokens": 30}, finish_reason="stop"
        ))

        response = await chain.generate(query="What are visiting hours?", top_k=4)

        assert response.langsmith_run_id is None
        assert response.langsmith_trace_id is None


# =============================================================================
# Tests for Tracing Failure Safety
# =============================================================================

class TestTracingFailureSafety:
    """Tests that LangSmith failures don't break core functionality."""

    @pytest.mark.asyncio
    async def test_rag_chain_works_when_langsmith_unavailable(self, monkeypatch):
        """Test that RAG chain works with invalid API key (simulates auth failure)."""
        from app.ai.rag.chain import RAGChain, RAGResponse
        from app.ai.providers.base import LLMResponse

        # Use monkeypatch to set environment variables
        monkeypatch.setenv("LANGSMITH_TRACING_V2", "true")
        monkeypatch.setenv("LANGSMITH_API_KEY", "invalid_key")

        chain = RAGChain()
        chain._retriever = MagicMock()
        chain._retriever.retrieve_with_scores.return_value = []
        chain._llm_provider = MagicMock()
        chain._llm_provider.model = "test-model"
        chain._llm_provider.generate_response = MagicMock(return_value=LLMResponse(
            content="Test answer", model="test-model",
            usage={"prompt_tokens": 10, "completion_tokens": 20}, finish_reason="stop"
        ))

        response = await chain.generate(query="Test query")
        assert isinstance(response, RAGResponse)
        assert response.answer == "Test answer"

    @pytest.mark.asyncio
    async def test_rag_chain_works_when_langsmith_post_fails(self, monkeypatch):
        """Test that RAG chain works when LangSmith RunTree.post() fails (network error).
        
        This simulates an actual LangSmith transport failure (network error when
        posting the run) while allowing the RAG chain itself to execute successfully.
        The @traceable decorator catches the post failure and logs it, but doesn't
        propagate it to the wrapped function.
        """
        import importlib
        import sys
        import app.ai.rag.chain as chain_module
        
        # Clear langsmith.utils from sys.modules to reset cached tracing state
        # (TestTracingDisabled fixture may have cached "disabled" state)
        for mod_name in list(sys.modules.keys()):
            if mod_name.startswith("langsmith"):
                del sys.modules[mod_name]
        
        # Set env vars BEFORE reloading so tracing is enabled when module loads
        monkeypatch.setenv("LANGSMITH_TRACING_V2", "true")
        monkeypatch.setenv("LANGSMITH_API_KEY", "test_key")
        
        # Reload module to pick up tracing config
        importlib.reload(chain_module)
        
        from app.ai.rag.chain import RAGChain, RAGResponse
        from app.ai.providers.base import LLMResponse
        import langsmith.run_trees as run_trees

        # Use a mock to track calls
        mock_post = MagicMock(side_effect=ConnectionError("LangSmith API unreachable"))
        
        # Use monkeypatch.setattr for automatic cleanup
        monkeypatch.setattr(run_trees.RunTree, "post", mock_post)
        
        # Import after reload
        from app.ai.rag.chain import RAGChain, RAGResponse
        from app.ai.providers.base import LLMResponse
        
        # Verify the generate method is actually traced (decorator was applied with tracing enabled)
        assert hasattr(RAGChain.generate, '__traceable_config__'), "generate should be traced"
        
        chain = RAGChain()
        chain._retriever = MagicMock()
        chain._retriever.retrieve_with_scores.return_value = []
        chain._llm_provider = MagicMock()
        chain._llm_provider.model = "test-model"
        chain._llm_provider.generate_response = MagicMock(return_value=LLMResponse(
            content="Test answer", model="test-model",
            usage={"prompt_tokens": 10, "completion_tokens": 20}, finish_reason="stop"
        ))

        response = await chain.generate(query="Test query")
        
        # Verify RAG chain still works
        assert isinstance(response, RAGResponse)
        assert response.answer == "Test answer"
        
        # Verify the post failure actually occurred (called for parent + child runs)
        assert mock_post.call_count >= 1, "RunTree.post should have been invoked at least once"


# =============================================================================
# Tests for Chat Route Integration
# =============================================================================

class TestChatRouteIntegration:
    @pytest.mark.asyncio
    async def test_chat_message_includes_langsmith_ids(self):
        from app.api.routes.chat.routes import ChatMessage
        msg = ChatMessage(id="msg1", role="assistant", content="Test response",
                          langsmith_run_id="run-123", langsmith_trace_id="trace-456")
        assert msg.langsmith_run_id == "run-123"
        assert msg.langsmith_trace_id == "trace-456"
        data = msg.model_dump()
        assert data["langsmith_run_id"] == "run-123"
        assert data["langsmith_trace_id"] == "trace-456"

    @pytest.mark.asyncio
    async def test_chat_message_works_without_langsmith_ids(self):
        from app.api.routes.chat.routes import ChatMessage
        msg = ChatMessage(id="msg1", role="assistant", content="Test response")
        assert msg.langsmith_run_id is None
        assert msg.langsmith_trace_id is None
        data = msg.model_dump()
        assert data["langsmith_run_id"] is None
        assert data["langsmith_trace_id"] is None


# =============================================================================
# Tests for SDK Callback Invocation Signatures
# =============================================================================

class TestSDKCallbackInvocation:
    @pytest.mark.asyncio
    async def test_rag_chain_process_inputs_signature_matches_sdk(self):
        from app.ai.tracing.config import rag_chain_process_inputs, rag_chain_process_outputs
        sdk_inputs = {"query": "Patient has fever", "conversation_history": [{"role": "user", "content": "Previous"}], "top_k": 4, "db": None}
        result = rag_chain_process_inputs(sdk_inputs)
        assert isinstance(result, dict)
        assert result["top_k"] == 4
        assert result["has_conversation_history"] is True
        assert result["history_length"] == 1
        assert "query" not in result
        assert "Patient has fever" not in str(result)

    @pytest.mark.asyncio
    async def test_retriever_process_inputs_signature_matches_sdk(self):
        from app.ai.tracing.config import retriever_process_inputs, retriever_process_outputs
        sdk_inputs = {"query": "patient diabetes", "top_k": 4, "min_score": 0.5, "filter_metadata": {"source": "medical"}}
        result = retriever_process_inputs(sdk_inputs)
        assert isinstance(result, dict)
        assert result["top_k"] == 4
        assert result["has_filter"] is True
        assert "query" not in result

    def test_faiss_process_inputs_signature_matches_sdk(self):
        from app.ai.tracing.config import faiss_search_process_inputs, faiss_search_process_outputs
        mock_index = MagicMock(); mock_index.ntotal = 1000
        sdk_inputs = {"query_embedding": [0.1] * 384, "top_k": 4, "filter_metadata": {"category": "medical"}, "self": type('obj', (object,), {'_index': mock_index})()}
        result = faiss_search_process_inputs(sdk_inputs)
        assert isinstance(result, dict)
        assert result["top_k"] == 4
        assert result["embedding_dim"] == 384
        assert result["has_filter"] is True
        assert result["index_size"] == 1000

    def test_groq_process_inputs_signature_matches_sdk(self):
        from app.ai.tracing.config import groq_llm_process_inputs, groq_llm_process_outputs
        mock_self = type('obj', (object,), {'model': "llama-3.1-8b-instant", 'temperature': 0.2, 'max_tokens': 2048})()
        sdk_inputs = {"messages": [MagicMock(role="user", content="Hello")], "system_prompt": "System prompt", "temperature": 0.5, "max_tokens": 1000, "self": mock_self}
        result = groq_llm_process_inputs(sdk_inputs)
        assert isinstance(result, dict)
        assert result["model"] == "llama-3.1-8b-instant"
        assert result["temperature"] == 0.5
        assert result["max_tokens"] == 1000
        assert result["message_count"] == 1
        assert result["has_system_prompt"] is True


# =============================================================================
# Verification: Sanitizers Configured on All Runnables
# =============================================================================

class TestSanitizerConfiguration:
    """Verify @traceable decorators have correct sanitizers attached.
    
    This test runs WITHOUT the local tracer to verify the real langsmith traceable config.
    """

    def test_all_runnables_have_sanitizers(self):
        # Import fresh modules without local tracer patching
        import importlib
        import app.ai.rag.chain as chain_module
        import app.ai.rag.retriever as retriever_module
        import app.ai.vectorstore.faiss_store as faiss_module
        import app.ai.providers.groq_provider as groq_module
        importlib.reload(chain_module)
        importlib.reload(retriever_module)
        importlib.reload(faiss_module)
        importlib.reload(groq_module)
        
        from app.ai.rag.chain import (
            _make_retriever_runnable, _make_is_medical_runnable,
            _make_detect_specialization_runnable, _make_build_prompt_runnable,
            _make_llm_runnable, _make_format_sources_runnable,
            _make_doctor_recommendations_runnable, _make_final_response_runnable,
        )
        from app.ai.rag.retriever import Retriever
        from app.ai.providers.base import BaseLLMProvider
        from app.ai.tracing.config import (
            retriever_runnable_process_inputs, retriever_runnable_process_outputs,
            is_medical_runnable_process_inputs, is_medical_runnable_process_outputs,
            detect_specialization_runnable_process_inputs, detect_specialization_runnable_process_outputs,
            build_prompt_runnable_process_inputs, build_prompt_runnable_process_outputs,
            llm_runnable_process_inputs, llm_runnable_process_outputs,
            format_sources_runnable_process_inputs, format_sources_runnable_process_outputs,
            doctor_recommendations_runnable_process_inputs, doctor_recommendations_runnable_process_outputs,
            final_response_runnable_process_inputs, final_response_runnable_process_outputs,
        )

        mock_retriever = MagicMock(spec=Retriever)
        mock_llm = MagicMock(spec=BaseLLMProvider); mock_llm.model = "test-model"

        checks = [
            (_make_retriever_runnable(mock_retriever), retriever_runnable_process_inputs, retriever_runnable_process_outputs),
            (_make_is_medical_runnable(), is_medical_runnable_process_inputs, is_medical_runnable_process_outputs),
            (_make_detect_specialization_runnable(), detect_specialization_runnable_process_inputs, detect_specialization_runnable_process_outputs),
            (_make_build_prompt_runnable(), build_prompt_runnable_process_inputs, build_prompt_runnable_process_outputs),
            (_make_llm_runnable(mock_llm), llm_runnable_process_inputs, llm_runnable_process_outputs),
            (_make_format_sources_runnable(), format_sources_runnable_process_inputs, format_sources_runnable_process_outputs),
            (_make_doctor_recommendations_runnable(), doctor_recommendations_runnable_process_inputs, doctor_recommendations_runnable_process_outputs),
            (_make_final_response_runnable(mock_llm), final_response_runnable_process_inputs, final_response_runnable_process_outputs),
        ]

        for runnable, expected_in, expected_out in checks:
            assert hasattr(runnable, '__traceable_config__'), f"{runnable} missing traceable config"
            config = runnable.__traceable_config__
            assert config['process_inputs'] is expected_in, f"Wrong process_inputs on {runnable}"
            assert config['process_outputs'] is expected_out, f"Wrong process_outputs on {runnable}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])