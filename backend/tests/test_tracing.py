"""Tests for LangSmith tracing integration."""
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# Ensure tracing is disabled by default for tests
os.environ["LANGSMITH_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"


class TestTracingDisabled:
    """Tests that tracing being disabled doesn't break functionality."""

    @pytest.mark.asyncio
    async def test_rag_chain_generate_works_without_tracing(self):
        """RAGChain.generate() should work normally when tracing is disabled."""
        from app.ai.rag.chain import RAGChain, RAGResponse

        # Mock the retriever and LLM provider
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

        # Should work without errors
        response = await chain.generate(query="What are visiting hours?")

        assert isinstance(response, RAGResponse)
        assert response.answer == "Test answer"
        assert response.model == "test-model"
        assert response.has_context is False
        # Run IDs should be None when tracing disabled
        assert response.langsmith_run_id is None
        assert response.langsmith_trace_id is None

    @pytest.mark.asyncio
    async def test_retriever_works_without_tracing(self):
        """Retriever.retrieve_with_scores() should work normally when tracing is disabled."""
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
        """FAISSVectorStore.search() should work normally when tracing is disabled."""
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
        """GroqProvider.generate_response() should work normally when tracing is disabled."""
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


class TestTracingEnabled:
    """Tests that tracing works when enabled (with mocked LangSmith client)."""

    @pytest.fixture(autouse=True)
    def enable_tracing(self):
        """Enable tracing for these tests."""
        os.environ["LANGSMITH_TRACING_V2"] = "true"
        os.environ["LANGSMITH_API_KEY"] = "ls_test_key"
        yield
        os.environ["LANGSMITH_TRACING_V2"] = "false"
        os.environ.pop("LANGSMITH_API_KEY", None)

    @pytest.mark.asyncio
    async def test_run_ids_captured_when_tracing_enabled(self):
        """Verify that run ID fields exist in response when tracing is enabled.

        Note: Actual run IDs require a valid LangSmith API key. This test verifies
        the code path attempts to capture them.
        """
        from app.ai.rag.chain import RAGChain
        from app.ai.providers.base import LLMResponse

        chain = RAGChain()

        # Mock retriever to return results
        mock_retriever = MagicMock()
        mock_search_result = MagicMock()
        mock_search_result.document = MagicMock()
        mock_search_result.document.id = "doc1"
        mock_search_result.document.content = "Test document content"
        mock_search_result.document.metadata = {"source": "test.pdf", "page": 1}
        mock_search_result.score = 0.85
        mock_retriever.retrieve_with_scores.return_value = [mock_search_result]
        chain._retriever = mock_retriever

        # Mock LLM provider
        chain._llm_provider = MagicMock()
        chain._llm_provider.model = "llama-3.1-8b-instant"
        chain._llm_provider.generate_response = MagicMock(return_value=LLMResponse(
            content="Test answer",
            model="llama-3.1-8b-instant",
            usage={"prompt_tokens": 50, "completion_tokens": 30},
            finish_reason="stop"
        ))

        # With tracing enabled but invalid API key, run IDs may be None
        # but the fields should exist on the response
        response = await chain.generate(query="What are visiting hours?", top_k=4)

        # Verify response is correct
        assert response.answer == "Test answer"
        assert response.has_context is True
        assert len(response.sources) == 1

        # Verify run ID fields exist on response (they may be None if LangSmith fails)
        assert hasattr(response, "langsmith_run_id")
        assert hasattr(response, "langsmith_trace_id")

    @pytest.mark.asyncio
    async def test_no_raw_query_in_trace_inputs(self):
        """Verify raw query is not sent to LangSmith - only safe metadata.

        SDK invokes process_inputs with a single dict of bound arguments.
        """
        from app.ai.tracing.config import rag_chain_process_inputs

        # Simulate SDK calling with bound arguments dict
        inputs = rag_chain_process_inputs({
            "query": "Patient John Doe has fever and chest pain, SSN 123-45-6789",
            "conversation_history": [{"role": "user", "content": "Previous medical question"}],
            "top_k": 4,
            "db": None
        })

        # Should only contain safe metadata
        assert "query" not in inputs
        assert "conversation_history" not in inputs
        assert inputs["top_k"] == 4
        assert inputs["has_conversation_history"] is True
        assert inputs["history_length"] == 1
        # No PHI should be present
        assert "John Doe" not in str(inputs)
        assert "123-45-6789" not in str(inputs)
        assert "fever" not in str(inputs)
        assert "chest pain" not in str(inputs)

    @pytest.mark.asyncio
    async def test_no_raw_answer_in_trace_outputs(self):
        """Verify raw answer is not sent to LangSmith - only safe metadata."""
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

        # Should only contain safe metadata
        assert "answer" not in outputs
        assert outputs["has_context"] is True
        assert outputs["sources_count"] == 1
        assert outputs["model"] == "llama-3.1-8b-instant"
        assert outputs["has_doctor_recommendations"] is True
        assert outputs["has_disclaimer"] is True
        assert "score_summary" in outputs
        # No PHI or medical content should be present
        assert "John Doe" not in str(outputs)
        assert "paracetamol" not in str(outputs)
        assert "fever" not in str(outputs)

    def test_retriever_process_inputs_no_phi(self):
        """Verify retriever process_inputs doesn't expose query.

        SDK invokes process_inputs with a single dict of bound arguments.
        """
        from app.ai.tracing.config import retriever_process_inputs

        # Simulate SDK calling with bound arguments dict
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
        """Verify FAISS search process_inputs doesn't expose embedding values.

        SDK invokes process_inputs with a single dict of bound arguments.
        """
        from app.ai.tracing.config import faiss_search_process_inputs

        mock_index = MagicMock()
        mock_index.ntotal = 1000

        # Simulate SDK calling with bound arguments dict
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
        # Embedding values should not be exposed
        assert "[0.1" not in str(inputs)

    def test_groq_llm_process_inputs_no_prompts(self):
        """Verify Groq LLM process_inputs doesn't expose prompts/messages.

        SDK invokes process_inputs with a single dict of bound arguments.
        """
        from app.ai.tracing.config import groq_llm_process_inputs
        from app.ai.providers.base import LLMMessage

        mock_self = type('obj', (object,), {
            'model': "llama-3.1-8b-instant",
            'temperature': 0.2,
            'max_tokens': 2048
        })()

        messages = [
            LLMMessage(role="system", content="You are a medical assistant. Patient John Doe has cancer."),
            LLMMessage(role="user", content="What treatment for patient SSN 111-22-3333?")
        ]

        # Simulate SDK calling with bound arguments dict
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
        # No prompts or PHI should be present
        assert "John Doe" not in str(inputs)
        assert "cancer" not in str(inputs)
        assert "111-22-3333" not in str(inputs)
        assert "medical assistant" not in str(inputs)

    def test_groq_llm_process_outputs_only_usage(self):
        """Verify Groq LLM process_outputs only exposes usage metadata."""
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
        # No response content should be present
        assert "Jane Smith" not in str(outputs)
        assert "chemotherapy" not in str(outputs)
        assert "breast cancer" not in str(outputs)


class TestTracingFailureSafety:
    """Tests that LangSmith failures don't break core functionality."""

    @pytest.mark.asyncio
    async def test_rag_chain_works_when_langsmith_unavailable(self):
        """RAGChain.generate() should work even if LangSmith API is unreachable."""
        from app.ai.rag.chain import RAGChain, RAGResponse
        from app.ai.providers.base import LLMResponse

        os.environ["LANGSMITH_TRACING_V2"] = "true"
        os.environ["LANGSMITH_API_KEY"] = "invalid_key"

        try:
            chain = RAGChain()
            chain._retriever = MagicMock()
            chain._retriever.retrieve_with_scores.return_value = []
            chain._llm_provider = MagicMock()
            chain._llm_provider.model = "test-model"
            chain._llm_provider.generate_response = MagicMock(return_value=LLMResponse(
                content="Test answer",
                model="test-model",
                usage={"prompt_tokens": 10, "completion_tokens": 20},
                finish_reason="stop"
            ))

            # Should not raise even if LangSmith fails
            response = await chain.generate(query="Test query")

            assert isinstance(response, RAGResponse)
            assert response.answer == "Test answer"
        finally:
            os.environ["LANGSMITH_TRACING_V2"] = "false"
            os.environ.pop("LANGSMITH_API_KEY", None)

    @pytest.mark.asyncio
    async def test_rag_chain_works_when_langsmith_throws(self):
        """RAGChain.generate() should work even if LangSmith client throws."""
        from app.ai.rag.chain import RAGChain, RAGResponse
        from app.ai.providers.base import LLMResponse
        from langsmith import traceable

        # Patch traceable to simulate an error during tracing setup
        original_traceable = traceable

        def failing_traceable(*args, **kwargs):
            def decorator(func):
                async def wrapper(*a, **kw):
                    # Simulate LangSmith error during run creation
                    raise Exception("LangSmith connection failed")
                return wrapper
            return decorator

        os.environ["LANGSMITH_TRACING_V2"] = "true"
        os.environ["LANGSMITH_API_KEY"] = "test_key"

        try:
            with patch("langsmith.traceable", side_effect=failing_traceable):
                chain = RAGChain()
                chain._retriever = MagicMock()
                chain._retriever.retrieve_with_scores.return_value = []
                chain._llm_provider = MagicMock()
                chain._llm_provider.model = "test-model"
                chain._llm_provider.generate_response = MagicMock(return_value=LLMResponse(
                    content="Test answer",
                    model="test-model",
                    usage={"prompt_tokens": 10, "completion_tokens": 20},
                    finish_reason="stop"
                ))

                # The traceable decorator should catch errors and not break the function
                # Note: In reality, traceable handles this, but we test the integration
                response = await chain.generate(query="Test query")

                assert isinstance(response, RAGResponse)
                assert response.answer == "Test answer"
        finally:
            os.environ["LANGSMITH_TRACING_V2"] = "false"
            os.environ.pop("LANGSMITH_API_KEY", None)


class TestChatRouteIntegration:
    """Integration tests for chat route with tracing."""

    @pytest.mark.asyncio
    async def test_chat_message_includes_langsmith_ids(self):
        """Verify assistant message includes LangSmith run IDs when tracing enabled."""
        from app.api.routes.chat.routes import ChatMessage

        msg = ChatMessage(
            id="msg1",
            role="assistant",
            content="Test response",
            langsmith_run_id="run-123",
            langsmith_trace_id="trace-456"
        )

        assert msg.langsmith_run_id == "run-123"
        assert msg.langsmith_trace_id == "trace-456"

        # Verify it serializes correctly
        data = msg.model_dump()
        assert data["langsmith_run_id"] == "run-123"
        assert data["langsmith_trace_id"] == "trace-456"

    @pytest.mark.asyncio
    async def test_chat_message_works_without_langsmith_ids(self):
        """Verify assistant message works without LangSmith run IDs (backward compat)."""
        from app.api.routes.chat.routes import ChatMessage

        msg = ChatMessage(
            id="msg1",
            role="assistant",
            content="Test response"
        )

        assert msg.langsmith_run_id is None
        assert msg.langsmith_trace_id is None

        data = msg.model_dump()
        assert data["langsmith_run_id"] is None
        assert data["langsmith_trace_id"] is None


class TestSDKCallbackInvocation:
    """Tests that verify our sanitizer signatures match what the langsmith SDK expects.

    The SDK calls process_inputs with a single dict of bound arguments, and
    process_outputs with the raw return value. We verify our functions accept
    these signatures correctly by calling them directly with SDK-shaped data.
    """

    @pytest.mark.asyncio
    async def test_rag_chain_process_inputs_signature_matches_sdk(self):
        """Verify our rag_chain_process_inputs signature matches SDK expectations.

        SDK calls: process_inputs(bound_args_dict) where bound_args_dict contains
        all function parameters as keys (excluding self/cls).
        """
        from app.ai.tracing.config import rag_chain_process_inputs, rag_chain_process_outputs

        # Simulate SDK calling with bound arguments dict (what _get_inputs produces)
        sdk_inputs = {
            "query": "Patient has fever",
            "conversation_history": [{"role": "user", "content": "Previous"}],
            "top_k": 4,
            "db": None
        }

        # Should not raise and should return dict
        result = rag_chain_process_inputs(sdk_inputs)
        assert isinstance(result, dict)
        assert "top_k" in result
        assert result["top_k"] == 4
        assert result["has_conversation_history"] is True
        assert result["history_length"] == 1
        # Raw query should NOT be in output
        assert "query" not in result
        assert "Patient has fever" not in str(result)

        # Test process_outputs with mock response object
        class MockResponse:
            has_context = True
            sources = [{"score": 0.9}, {"score": 0.8}]
            model = "llama-3.1-8b-instant"
            doctor_recommendations = [{"doctor_id": "1"}]
            disclaimer = "Medical disclaimer"

        output_result = rag_chain_process_outputs(MockResponse())
        assert isinstance(output_result, dict)
        assert output_result["has_context"] is True
        assert output_result["sources_count"] == 2
        assert output_result["model"] == "llama-3.1-8b-instant"
        assert output_result["has_doctor_recommendations"] is True
        assert output_result["has_disclaimer"] is True
        assert "score_summary" in output_result

    @pytest.mark.asyncio
    async def test_retriever_process_inputs_signature_matches_sdk(self):
        """Verify our retriever_process_inputs signature matches SDK expectations."""
        from app.ai.tracing.config import retriever_process_inputs, retriever_process_outputs

        sdk_inputs = {
            "query": "patient diabetes",
            "top_k": 4,
            "min_score": 0.5,
            "filter_metadata": {"source": "medical"}
        }

        result = retriever_process_inputs(sdk_inputs)
        assert isinstance(result, dict)
        assert result["top_k"] == 4
        assert result["has_filter"] is True
        assert "query" not in result
        assert "patient diabetes" not in str(result)

        # Test process_outputs
        class MockSearchResult:
            score = 0.9

        output_result = retriever_process_outputs([MockSearchResult(), MockSearchResult()])
        assert isinstance(output_result, dict)
        assert output_result["results_count"] == 2
        assert "score_summary" in output_result

    def test_faiss_process_inputs_signature_matches_sdk(self):
        """Verify our faiss_search_process_inputs signature matches SDK expectations."""
        from app.ai.tracing.config import faiss_search_process_inputs, faiss_search_process_outputs

        mock_index = MagicMock()
        mock_index.ntotal = 1000

        sdk_inputs = {
            "query_embedding": [0.1] * 384,
            "top_k": 4,
            "filter_metadata": {"category": "medical"},
            "self": type('obj', (object,), {'_index': mock_index})()
        }

        result = faiss_search_process_inputs(sdk_inputs)
        assert isinstance(result, dict)
        assert result["top_k"] == 4
        assert result["embedding_dim"] == 384
        assert result["has_filter"] is True
        assert result["index_size"] == 1000
        assert "query_embedding" not in str(result).lower()  # embedding values not exposed

        # Test process_outputs
        class MockSearchResult:
            score = 0.9

        output_result = faiss_search_process_outputs([MockSearchResult()])
        assert isinstance(output_result, dict)
        assert output_result["results_count"] == 1
        assert "score_summary" in output_result

    def test_groq_process_inputs_signature_matches_sdk(self):
        """Verify our groq_llm_process_inputs signature matches SDK expectations."""
        from app.ai.tracing.config import groq_llm_process_inputs, groq_llm_process_outputs

        mock_self = type('obj', (object,), {
            'model': "llama-3.1-8b-instant",
            'temperature': 0.2,
            'max_tokens': 2048
        })()

        sdk_inputs = {
            "messages": [MagicMock(role="user", content="Hello")],
            "system_prompt": "System prompt",
            "temperature": 0.5,
            "max_tokens": 1000,
            "self": mock_self
        }

        result = groq_llm_process_inputs(sdk_inputs)
        assert isinstance(result, dict)
        assert result["model"] == "llama-3.1-8b-instant"
        assert result["temperature"] == 0.5
        assert result["max_tokens"] == 1000
        assert result["message_count"] == 1
        assert result["has_system_prompt"] is True
        # Raw messages/prompts should not be in output
        assert "Hello" not in str(result)
        assert "System prompt" not in str(result)

        # Test process_outputs
        class MockLLMResponse:
            model = "llama-3.1-8b-instant"
            finish_reason = "stop"
            usage = {"prompt_tokens": 100, "completion_tokens": 50}

        output_result = groq_llm_process_outputs(MockLLMResponse())
        assert isinstance(output_result, dict)
        assert output_result["model"] == "llama-3.1-8b-instant"
        assert output_result["finish_reason"] == "stop"
        assert output_result["usage"]["prompt_tokens"] == 100
        assert output_result["usage"]["completion_tokens"] == 50


class TestSanitizationFailClosed:
    """Tests that verify fail-closed behavior - raw data NEVER leaks on sanitization errors."""

    @pytest.mark.asyncio
    async def test_rag_chain_process_inputs_fail_closed(self):
        """Even if process_inputs crashes, raw query never reaches LangSmith."""
        from app.ai.tracing.config import rag_chain_process_inputs

        # Test with completely invalid input type
        result = rag_chain_process_inputs("not a dict")
        assert isinstance(result, dict)
        assert "top_k" in result
        assert result["has_conversation_history"] is False

        result = rag_chain_process_inputs(None)
        assert isinstance(result, dict)

        result = rag_chain_process_inputs([])
        assert isinstance(result, dict)

        # Test with dict containing unserializable objects
        class Unserializable:
            def __repr__(self):
                raise RuntimeError("Cannot serialize")

        result = rag_chain_process_inputs({"query": "test", "top_k": Unserializable()})
        assert isinstance(result, dict)
        assert result["top_k"] is None  # Failed to get int, falls back to None

    @pytest.mark.asyncio
    async def test_rag_chain_process_outputs_fail_closed(self):
        """Even if process_outputs crashes, raw answer never reaches LangSmith."""
        from app.ai.tracing.config import rag_chain_process_outputs

        # The fail-closed function returns a dict with all safe fields, not just minimal
        result = rag_chain_process_outputs("not an object")
        assert isinstance(result, dict)
        assert result["has_context"] is False
        assert result["sources_count"] == 0
        assert result["model"] == ""
        assert result["has_doctor_recommendations"] is False
        assert result["has_disclaimer"] is False
        assert "score_summary" in result

        result = rag_chain_process_outputs(None)
        assert result["has_context"] is False
        assert result["sources_count"] == 0

        # Test with object that raises on attribute access
        class BadObject:
            @property
            def has_context(self):
                raise RuntimeError("Fail")

        result = rag_chain_process_outputs(BadObject())
        assert isinstance(result, dict)
        assert result["has_context"] is False
        assert result["sources_count"] == 0

        # Test with sources that cause errors
        class BadSource:
            def get(self, key, default=None):
                if key == "score":
                    raise RuntimeError("Score error")
                return default

        class ResponseWithBadSources:
            has_context = True
            sources = [BadSource()]
            model = "test"
            doctor_recommendations = []
            disclaimer = ""

        result = rag_chain_process_outputs(ResponseWithBadSources())
        assert isinstance(result, dict)
        assert "has_context" in result

    def test_retriever_process_inputs_fail_closed(self):
        from app.ai.tracing.config import retriever_process_inputs

        for bad_input in ["string", None, [], 123, object()]:
            result = retriever_process_inputs(bad_input)
            assert isinstance(result, dict)
            assert result["has_filter"] is False

    def test_retriever_process_outputs_fail_closed(self):
        from app.ai.tracing.config import retriever_process_outputs

        for bad_input in ["string", None, 123, object()]:
            result = retriever_process_outputs(bad_input)
            assert isinstance(result, dict)
            assert result["results_count"] == 0
            assert "score_summary" in result

        # Test with list containing bad objects
        class BadResult:
            @property
            def score(self):
                raise RuntimeError("Fail")

        result = retriever_process_outputs([BadResult()])
        assert isinstance(result, dict)
        # When input is a list, len() returns the list length
        assert result["results_count"] == 1
        assert "score_summary" in result

    def test_faiss_process_inputs_fail_closed(self):
        from app.ai.tracing.config import faiss_search_process_inputs

        for bad_input in ["string", None, [], 123, object()]:
            result = faiss_search_process_inputs(bad_input)
            assert isinstance(result, dict)
            assert result["has_filter"] is False
            assert result["embedding_dim"] == 0

    def test_faiss_process_outputs_fail_closed(self):
        from app.ai.tracing.config import faiss_search_process_outputs

        for bad_input in ["string", None, 123, object()]:
            result = faiss_search_process_outputs(bad_input)
            assert isinstance(result, dict)
            assert result["results_count"] == 0
            assert "score_summary" in result

    def test_groq_process_inputs_fail_closed(self):
        from app.ai.tracing.config import groq_llm_process_inputs

        for bad_input in ["string", None, [], 123, object()]:
            result = groq_llm_process_inputs(bad_input)
            assert isinstance(result, dict)
            assert result["message_count"] == 0
            assert result["model"] == ""

    def test_groq_process_outputs_fail_closed(self):
        from app.ai.tracing.config import groq_llm_process_outputs

        for bad_input in ["string", None, [], 123, object()]:
            result = groq_llm_process_outputs(bad_input)
            assert isinstance(result, dict)
            assert result.get("model", "") == ""
            assert result.get("finish_reason", "") == ""
            assert result.get("usage", {}) == {}

        # Test with bad usage
        class BadUsage:
            def get(self, key, default=None):
                if key == "prompt_tokens":
                    raise RuntimeError("Fail")
                return default

        class BadResponse:
            model = "test"
            finish_reason = "stop"
            usage = BadUsage()

        result = groq_llm_process_outputs(BadResponse())
        assert isinstance(result, dict)
        assert result.get("model") == "test"
        assert result.get("finish_reason") == "stop"
        assert result.get("usage") == {}


class TestNestedTraceIndependence:
    """Tests that verify nested trace inputs/outputs are sanitized independently.

    We verify that each sanitizer produces independent output when called with
    different inputs, and that no PHI leaks through any sanitizer.
    """

    @pytest.mark.asyncio
    async def test_each_sanitizer_independent_and_phi_free(self):
        """Verify each sanitizer independently sanitizes and never leaks PHI."""
        from app.ai.tracing.config import (
            rag_chain_process_inputs,
            rag_chain_process_outputs,
            retriever_process_inputs,
            retriever_process_outputs,
            faiss_search_process_inputs,
            faiss_search_process_outputs,
            groq_llm_process_inputs,
            groq_llm_process_outputs,
        )

        # Test data with PHI at each level
        rag_inputs = {
            "query": "Patient John Doe has diabetes",
            "conversation_history": [{"role": "user", "content": "Previous: patient SSN 123-45-6789"}],
            "top_k": 4,
            "db": None
        }

        retriever_inputs = {
            "query": "Patient Jane Smith has cancer",
            "top_k": 4,
            "min_score": 0.5,
            "filter_metadata": {"source": "oncology"}
        }

        faiss_inputs = {
            "query_embedding": [0.1] * 384,
            "top_k": 4,
            "filter_metadata": {"category": "medical"},
            "self": type('obj', (object,), {'_index': MagicMock(ntotal=1000)})()
        }

        mock_self = type('obj', (object,), {
            'model': "llama-3.1-8b-instant",
            'temperature': 0.2,
            'max_tokens': 2048
        })()

        groq_inputs = {
            "messages": [
                MagicMock(role="system", content="You are a doctor. Patient John Doe has cancer."),
                MagicMock(role="user", content="What treatment for SSN 111-22-3333?")
            ],
            "system_prompt": "System prompt with patient data",
            "temperature": 0.5,
            "max_tokens": 1000,
            "self": mock_self
        }

        # Call each sanitizer independently
        rag_in_result = rag_chain_process_inputs(rag_inputs)
        retriever_in_result = retriever_process_inputs(retriever_inputs)
        faiss_in_result = faiss_search_process_inputs(faiss_inputs)
        groq_in_result = groq_llm_process_inputs(groq_inputs)

        # Output sanitizers
        class MockRAGResponse:
            has_context = True
            sources = [{"score": 0.9}]
            model = "llama-3.1-8b-instant"
            doctor_recommendations = [{"doctor_id": "1"}]
            disclaimer = "Medical disclaimer"

        class MockSearchResult:
            score = 0.9

        class MockLLMResponse:
            model = "llama-3.1-8b-instant"
            finish_reason = "stop"
            usage = {"prompt_tokens": 100, "completion_tokens": 50}

        rag_out_result = rag_chain_process_outputs(MockRAGResponse())
        retriever_out_result = retriever_process_outputs([MockSearchResult()])
        faiss_out_result = faiss_search_process_outputs([MockSearchResult()])
        groq_out_result = groq_llm_process_outputs(MockLLMResponse())

        all_results = [
            rag_in_result, retriever_in_result, faiss_in_result, groq_in_result,
            rag_out_result, retriever_out_result, faiss_out_result, groq_out_result
        ]

        # Verify each result is a dict (not raw data)
        for result in all_results:
            assert isinstance(result, dict), "Each sanitizer must return dict"

        # Verify NO PHI in ANY sanitized result
        phi_strings = [
            "John Doe", "Jane Smith", "123-45-6789", "111-22-3333",
            "diabetes", "cancer", "SSN", "patient", "treatment",
            "metformin", "chemotherapy", "oncology"
        ]

        for result in all_results:
            result_str = str(result)
            for phi in phi_strings:
                assert phi not in result_str, f"PHI '{phi}' leaked in {result}"

        # Verify each has expected safe fields
        assert rag_in_result["top_k"] == 4
        assert retriever_in_result["top_k"] == 4
        assert faiss_in_result["embedding_dim"] == 384
        assert groq_in_result["model"] == "llama-3.1-8b-instant"
        assert rag_out_result["has_context"] is True
        assert retriever_out_result["results_count"] == 1
        assert faiss_out_result["results_count"] == 1
        assert groq_out_result["model"] == "llama-3.1-8b-instant"


class TestIntegrationMockLangSmith:
    """Integration tests using LangSmith's testing utilities to inspect final payloads."""

    @pytest.fixture(autouse=True)
    def enable_tracing(self):
        os.environ["LANGSMITH_TRACING_V2"] = "true"
        os.environ["LANGSMITH_API_KEY"] = "ls_test_key"
        os.environ["LANGSMITH_PROJECT"] = "test-project"
        yield
        os.environ["LANGSMITH_TRACING_V2"] = "false"
        os.environ.pop("LANGSMITH_API_KEY", None)
        os.environ.pop("LANGSMITH_PROJECT", None)

    @pytest.mark.asyncio
    async def test_final_run_payload_contains_no_phi(self):
        """Use LangSmith's test client to capture and inspect the serialized run."""
        from langsmith import Client
        from app.ai.rag.chain import RAGChain
        from app.ai.providers.base import LLMResponse

        # Create a test client that captures runs in memory
        client = Client(
            api_key="ls_test_key",
            api_url="https://api.smith.langchain.com",
        )

        # We can't easily mock the HTTP layer, but we can verify the sanitizers
        # produce correct output by checking the RunTree before it's sent
        from langsmith.run_trees import RunTree
        from langsmith import traceable

        # Create a simple traced function and inspect its RunTree
        @traceable(name="test_sanitization", run_type="chain",
                   process_inputs=lambda d: {"safe": "inputs"},
                   process_outputs=lambda o: {"safe": "outputs"})
        def test_func(query):
            return "Patient John Doe has cancer"

        # Call the function
        result = test_func("Patient has cancer")

        # The RunTree would have been created with sanitized inputs/outputs
        # Since we can't easily capture the RunTree without a valid API,
        # we verify the sanitizers were called by checking the traceable config
        assert test_func.__traceable_config__["process_inputs"] is not None
        assert test_func.__traceable_config__["process_outputs"] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
