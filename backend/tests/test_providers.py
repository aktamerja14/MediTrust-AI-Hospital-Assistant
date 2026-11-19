"""Tests for LLM Provider Factory and Provider Implementations.

Tests provider selection, initialization, and response generation with mocked API calls.
"""
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Any
import httpx

from app.ai.factory import ProviderFactory
from app.ai.providers.base import LLMMessage, LLMResponse
from app.ai.providers.groq_provider import GroqProvider
from app.ai.providers.gemini_provider import GeminiProvider
from app.ai.providers.mock_provider import MockProvider
from app.ai.providers.openrouter_provider import OpenRouterProvider
from app.ai.providers.base import (
    LLMAuthenticationError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMModelNotFoundError,
    LLMProviderUnavailableError,
)
from app.core.config import Settings


def _create_test_settings(**overrides) -> Settings:
    """Create a Settings instance with test values."""
    defaults = {
        "LLM_PROVIDER": "mock",
        "LLM_MODEL": "mock-model",
        "LLM_API_KEY": "mock-key",
        "LLM_BASE_URL": "",
        "LLM_TEMPERATURE": 0.2,
        "LLM_MAX_TOKENS": 2048,
        "LLM_TIMEOUT": 60,
        "MONGODB_URL": "mongodb://localhost:27017",
        "DATABASE_NAME": "meditrust_test",
        "JWT_SECRET_KEY": "test-secret",
        "ENVIRONMENT": "testing",
    }
    defaults.update(overrides)
    return Settings(**defaults)


def _patch_settings(**overrides):
    """Context manager to patch settings used by factory."""
    test_settings = _create_test_settings(**overrides)
    return patch("app.ai.factory.settings", test_settings)


class TestProviderFactory:
    """Tests for ProviderFactory provider selection and instantiation."""

    def test_list_available_providers(self):
        """Test that all expected providers are registered."""
        providers = ProviderFactory.list_available()
        assert "groq" in providers
        assert "gemini" in providers
        assert "mock" in providers
        assert "openrouter" in providers

    def test_create_groq_provider(self):
        """Test creating Groq provider with correct configuration."""
        ProviderFactory.reset()
        with _patch_settings(
            LLM_PROVIDER="groq",
            LLM_MODEL="openai/gpt-oss-120b",
            LLM_API_KEY="gsk_test_key",
            LLM_BASE_URL="https://api.groq.com/openai/v1",
        ):
            provider = ProviderFactory.create("groq")

            assert provider.provider_name == "groq"
            assert provider.model == "openai/gpt-oss-120b"
            assert provider.base_url == "https://api.groq.com/openai/v1"
            assert provider.temperature == 0.2
            assert provider.max_tokens == 2048
            assert provider.timeout == 60

    def test_create_gemini_provider(self):
        """Test creating Gemini provider with correct configuration."""
        ProviderFactory.reset()
        with _patch_settings(
            LLM_PROVIDER="gemini",
            LLM_MODEL="gemini-3.8-flash",
            LLM_API_KEY="gemini_test_key",
            LLM_BASE_URL="",
        ):
            provider = ProviderFactory.create("gemini")

            assert provider.provider_name == "gemini"
            assert provider.model == "gemini-3.8-flash"
            assert provider.base_url == ""

    def test_create_mock_provider(self):
        """Test creating Mock provider (no API key required)."""
        ProviderFactory.reset()
        with _patch_settings(
            LLM_PROVIDER="mock",
            LLM_MODEL="mock-model",
            LLM_API_KEY="",
            LLM_BASE_URL="",
        ):
            provider = ProviderFactory.create("mock")

            assert provider.provider_name == "mock"
            assert provider.model == "mock-model"

    def test_create_openrouter_provider(self):
        """Test creating OpenRouter provider with correct configuration."""
        ProviderFactory.reset()
        with _patch_settings(
            LLM_PROVIDER="openrouter",
            LLM_MODEL="meta-llama/llama-3.1-8b-instruct:free",
            LLM_API_KEY="or_test_key",
            LLM_BASE_URL="https://openrouter.ai/api/v1",
        ):
            provider = ProviderFactory.create("openrouter")

            assert provider.provider_name == "openrouter"
            assert provider.model == "meta-llama/llama-3.1-8b-instruct:free"
            assert provider.base_url == "https://openrouter.ai/api/v1"

    def test_unknown_provider_raises_error(self):
        """Test that unknown provider raises ValueError."""
        ProviderFactory.reset()
        with _patch_settings(LLM_PROVIDER="unknown", LLM_API_KEY="test"):
            with pytest.raises(ValueError, match="Unknown LLM provider: 'unknown'"):
                ProviderFactory.create("unknown")

    def test_missing_api_key_raises_auth_error(self):
        """Test that missing API key raises LLMAuthenticationError for non-mock providers."""
        ProviderFactory.reset()
        with _patch_settings(LLM_PROVIDER="groq", LLM_API_KEY=""):
            with pytest.raises(LLMAuthenticationError, match="LLM_API_KEY is required"):
                ProviderFactory.create("groq")

    def test_singleton_instance(self):
        """Test that get_instance returns singleton."""
        ProviderFactory.reset()
        with _patch_settings(LLM_PROVIDER="mock", LLM_MODEL="mock-model", LLM_API_KEY="mock-key"):
            instance1 = ProviderFactory.get_instance()
            instance2 = ProviderFactory.get_instance()
            assert instance1 is instance2

    def test_reset_clears_singleton(self):
        """Test that reset clears the singleton instance."""
        ProviderFactory.reset()
        with _patch_settings(LLM_PROVIDER="mock", LLM_MODEL="mock-model", LLM_API_KEY="mock-key"):
            instance1 = ProviderFactory.get_instance()
            ProviderFactory.reset()
            instance2 = ProviderFactory.get_instance()
            assert instance1 is not instance2


class TestMockProvider:
    """Tests for MockProvider with canned responses."""

    def test_mock_provider_initialization(self):
        """Test MockProvider initializes without API key."""
        provider = MockProvider(api_key="", model="test-model")
        assert provider.provider_name == "mock"
        assert provider.model == "test-model"

    def test_mock_provider_generate_response(self):
        """Test MockProvider returns canned responses."""
        provider = MockProvider(api_key="mock-key", model="mock-model")

        # Test hypertension response
        messages = [LLMMessage(role="user", content="What are symptoms of high blood pressure?")]
        response = provider.generate_response(messages)
        assert isinstance(response, LLMResponse)
        assert "hypertension" in response.content.lower() or "blood pressure" in response.content.lower()
        assert response.model == "mock-model"
        assert response.usage is not None
        assert response.finish_reason == "stop"

    def test_mock_provider_greeting_response(self):
        """Test MockProvider greeting response."""
        provider = MockProvider(api_key="mock-key", model="mock-model")

        messages = [LLMMessage(role="user", content="Hello")]
        response = provider.generate_response(messages)
        assert "hello" in response.content.lower() or "assistant" in response.content.lower()

    def test_mock_provider_upload_response(self):
        """Test MockProvider document upload response."""
        provider = MockProvider(api_key="mock-key", model="mock-model")

        messages = [LLMMessage(role="user", content="How do I upload a document?")]
        response = provider.generate_response(messages)
        assert "upload" in response.content.lower() or "document" in response.content.lower()

    def test_mock_provider_default_response(self):
        """Test MockProvider default fallback response."""
        provider = MockProvider(api_key="mock-key", model="mock-model")

        messages = [LLMMessage(role="user", content="Random unknown query xyz123")]
        response = provider.generate_response(messages)
        assert "Mock response to: Random unknown query" in response.content


class TestGroqProvider:
    """Tests for GroqProvider with mocked OpenAI client."""

    def test_groq_provider_initialization(self):
        """Test GroqProvider initializes with correct defaults."""
        provider = GroqProvider(
            api_key="gsk_test_key",
            model="openai/gpt-oss-120b",
            base_url="https://api.groq.com/openai/v1",
        )
        assert provider.provider_name == "groq"
        assert provider.model == "openai/gpt-oss-120b"
        assert provider.base_url == "https://api.groq.com/openai/v1"

    def test_groq_provider_default_base_url(self):
        """Test GroqProvider uses default base URL when not provided."""
        provider = GroqProvider(api_key="gsk_test_key", model="test-model")
        assert provider.base_url == "https://api.groq.com/openai/v1"

    @patch("app.ai.providers.groq_provider.OpenAI")
    def test_groq_provider_generate_response_success(self, mock_openai_class):
        """Test GroqProvider generates response successfully."""
        # Setup mock
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_choice = MagicMock()
        mock_choice.message.content = "Groq response content"
        mock_choice.finish_reason = "stop"

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 20

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.model = "openai/gpt-oss-120b"
        mock_response.usage = mock_usage

        mock_client.chat.completions.create.return_value = mock_response

        # Test
        provider = GroqProvider(api_key="gsk_test_key", model="openai/gpt-oss-120b")
        messages = [LLMMessage(role="user", content="Test query")]
        response = provider.generate_response(messages, system_prompt="System prompt")

        # Verify
        assert isinstance(response, LLMResponse)
        assert response.content == "Groq response content"
        assert response.model == "openai/gpt-oss-120b"
        assert response.usage["prompt_tokens"] == 10
        assert response.usage["completion_tokens"] == 20
        assert response.finish_reason == "stop"

        # Verify API call
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["model"] == "openai/gpt-oss-120b"
        assert len(call_args.kwargs["messages"]) == 2  # system + user

    @patch("app.ai.providers.groq_provider.OpenAI")
    def test_groq_provider_empty_response_raises_error(self, mock_openai_class):
        """Test GroqProvider raises error on empty response."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_choice = MagicMock()
        mock_choice.message.content = ""

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.model = "test-model"
        mock_response.usage = None

        mock_client.chat.completions.create.return_value = mock_response

        provider = GroqProvider(api_key="gsk_test_key", model="test-model")
        messages = [LLMMessage(role="user", content="Test")]

        with pytest.raises(LLMProviderUnavailableError, match="Empty response from Groq"):
            provider.generate_response(messages)

    @patch("app.ai.providers.groq_provider.OpenAI")
    def test_groq_provider_authentication_error(self, mock_openai_class):
        """Test GroqProvider handles authentication error."""
        from openai import AuthenticationError as OpenAIAuthError

        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.side_effect = OpenAIAuthError(
            "Invalid API key", response=MagicMock(status_code=401), body={}
        )

        provider = GroqProvider(api_key="invalid_key", model="test-model")
        messages = [LLMMessage(role="user", content="Test")]

        with pytest.raises(LLMAuthenticationError, match="Invalid API key"):
            provider.generate_response(messages)

    @patch("app.ai.providers.groq_provider.OpenAI")
    def test_groq_provider_rate_limit_error(self, mock_openai_class):
        """Test GroqProvider handles rate limit error."""
        from openai import RateLimitError as OpenAIRateLimitError

        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.side_effect = OpenAIRateLimitError(
            "Rate limit", response=MagicMock(status_code=429), body={}
        )

        provider = GroqProvider(api_key="gsk_test_key", model="test-model")
        messages = [LLMMessage(role="user", content="Test")]

        with pytest.raises(LLMRateLimitError, match="Rate limit exceeded"):
            provider.generate_response(messages)

    @patch("app.ai.providers.groq_provider.OpenAI")
    def test_groq_provider_timeout_error(self, mock_openai_class):
        """Test GroqProvider handles timeout error."""
        from openai import APITimeoutError as OpenAITimeout

        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.side_effect = OpenAITimeout("Timeout")

        provider = GroqProvider(api_key="gsk_test_key", model="test-model")
        messages = [LLMMessage(role="user", content="Test")]

        with pytest.raises(LLMTimeoutError, match="Request timed out"):
            provider.generate_response(messages)

    @patch("app.ai.providers.groq_provider.OpenAI")
    def test_groq_provider_model_not_found_error(self, mock_openai_class):
        """Test GroqProvider handles model not found error."""
        from openai import NotFoundError as OpenAINotFoundError

        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.side_effect = OpenAINotFoundError(
            "Model not found", response=MagicMock(status_code=404), body={}
        )

        provider = GroqProvider(api_key="gsk_test_key", model="invalid-model")
        messages = [LLMMessage(role="user", content="Test")]

        with pytest.raises(LLMModelNotFoundError, match="Model 'invalid-model' not found"):
            provider.generate_response(messages)


class TestGeminiProvider:
    """Tests for GeminiProvider with mocked google-genai client."""

    def test_gemini_provider_initialization(self):
        """Test GeminiProvider initializes correctly."""
        provider = GeminiProvider(
            api_key="gemini_test_key",
            model="gemini-3.8-flash",
        )
        assert provider.provider_name == "gemini"
        assert provider.model == "gemini-3.8-flash"

    @patch("app.ai.providers.gemini_provider.genai.Client")
    def test_gemini_provider_generate_response_success(self, mock_genai_client):
        """Test GeminiProvider generates response successfully."""
        # Setup mock
        mock_client_instance = MagicMock()
        mock_genai_client.return_value = mock_client_instance

        mock_usage_metadata = MagicMock()
        mock_usage_metadata.prompt_token_count = 10
        mock_usage_metadata.candidates_token_count = 20

        mock_candidate = MagicMock()
        mock_candidate.finish_reason = "STOP"

        mock_response = MagicMock()
        mock_response.text = "Gemini response content"
        mock_response.usage_metadata = mock_usage_metadata
        mock_response.candidates = [mock_candidate]

        mock_client_instance.models.generate_content.return_value = mock_response

        # Test
        provider = GeminiProvider(api_key="gemini_test_key", model="gemini-3.8-flash")
        messages = [LLMMessage(role="user", content="Test query")]
        response = provider.generate_response(messages, system_prompt="System prompt")

        # Verify
        assert isinstance(response, LLMResponse)
        assert response.content == "Gemini response content"
        assert response.model == "gemini-3.8-flash"
        assert response.usage["prompt_tokens"] == 10
        assert response.usage["completion_tokens"] == 20
        assert response.finish_reason == "STOP"

        # Verify API call
        mock_client_instance.models.generate_content.assert_called_once()
        call_args = mock_client_instance.models.generate_content.call_args
        assert call_args.kwargs["model"] == "gemini-3.8-flash"
        assert "contents" in call_args.kwargs
        assert "config" in call_args.kwargs

    @patch("app.ai.providers.gemini_provider.genai.Client")
    def test_gemini_provider_empty_response_raises_error(self, mock_genai_client):
        """Test GeminiProvider raises error on empty response."""
        mock_client_instance = MagicMock()
        mock_genai_client.return_value = mock_client_instance

        mock_response = MagicMock()
        mock_response.text = ""
        mock_response.usage_metadata = None
        mock_response.candidates = []

        mock_client_instance.models.generate_content.return_value = mock_response

        provider = GeminiProvider(api_key="gemini_test_key", model="gemini-3.8-flash")
        messages = [LLMMessage(role="user", content="Test")]

        with pytest.raises(LLMProviderUnavailableError, match="Empty response from Gemini"):
            provider.generate_response(messages)

    @patch("app.ai.providers.gemini_provider.genai.Client")
    def test_gemini_provider_authentication_error(self, mock_genai_client):
        """Test GeminiProvider handles authentication error."""
        from google.genai import errors as genai_errors
        import httpx

        mock_client_instance = MagicMock()
        mock_genai_client.return_value = mock_client_instance

        response = httpx.Response(401, json={"error": {"message": "API_KEY_INVALID: Invalid API key"}})
        error = genai_errors.APIError(
            code=401,
            response_json={"error": {"message": "API_KEY_INVALID: Invalid API key"}},
            response=response,
        )
        mock_client_instance.models.generate_content.side_effect = error

        provider = GeminiProvider(api_key="invalid_key", model="gemini-3.8-flash")
        messages = [LLMMessage(role="user", content="Test")]

        with pytest.raises(LLMAuthenticationError, match="Invalid API key or insufficient permissions"):
            provider.generate_response(messages)

    @patch("app.ai.providers.gemini_provider.genai.Client")
    def test_gemini_provider_rate_limit_error(self, mock_genai_client):
        """Test GeminiProvider handles rate limit error."""
        from google.genai import errors as genai_errors
        import httpx

        mock_client_instance = MagicMock()
        mock_genai_client.return_value = mock_client_instance

        response = httpx.Response(429, json={"error": {"message": "RESOURCE_EXHAUSTED: Rate limit exceeded"}})
        error = genai_errors.APIError(
            code=429,
            response_json={"error": {"message": "RESOURCE_EXHAUSTED: Rate limit exceeded"}},
            response=response,
        )
        mock_client_instance.models.generate_content.side_effect = error

        provider = GeminiProvider(api_key="gemini_test_key", model="gemini-3.8-flash")
        messages = [LLMMessage(role="user", content="Test")]

        with pytest.raises(LLMRateLimitError, match="Rate limit exceeded"):
            provider.generate_response(messages)

    @patch("app.ai.providers.gemini_provider.genai.Client")
    def test_gemini_provider_timeout_error(self, mock_genai_client):
        """Test GeminiProvider handles timeout error."""
        from google.genai import errors as genai_errors
        import httpx

        mock_client_instance = MagicMock()
        mock_genai_client.return_value = mock_client_instance

        response = httpx.Response(504, json={"error": {"message": "DEADLINE_EXCEEDED: Request timeout"}})
        error = genai_errors.APIError(
            code=504,
            response_json={"error": {"message": "DEADLINE_EXCEEDED: Request timeout"}},
            response=response,
        )
        mock_client_instance.models.generate_content.side_effect = error

        provider = GeminiProvider(api_key="gemini_test_key", model="gemini-3.8-flash")
        messages = [LLMMessage(role="user", content="Test")]

        with pytest.raises(LLMTimeoutError, match="Request timed out"):
            provider.generate_response(messages)

    @patch("app.ai.providers.gemini_provider.genai.Client")
    def test_gemini_provider_model_not_found_error(self, mock_genai_client):
        """Test GeminiProvider handles model not found error."""
        from google.genai import errors as genai_errors
        import httpx

        mock_client_instance = MagicMock()
        mock_genai_client.return_value = mock_client_instance

        response = httpx.Response(404, json={"error": {"message": "NOT_FOUND: Model 'invalid-model' not found"}})
        error = genai_errors.APIError(
            code=404,
            response_json={"error": {"message": "NOT_FOUND: Model 'invalid-model' not found"}},
            response=response,
        )
        mock_client_instance.models.generate_content.side_effect = error

        provider = GeminiProvider(api_key="gemini_test_key", model="invalid-model")
        messages = [LLMMessage(role="user", content="Test")]

        with pytest.raises(LLMModelNotFoundError, match="Model 'invalid-model' not found"):
            provider.generate_response(messages)


class TestOpenRouterProvider:
    """Tests for OpenRouterProvider with mocked OpenAI client."""

    def test_openrouter_provider_initialization(self):
        """Test OpenRouterProvider initializes with correct defaults."""
        provider = OpenRouterProvider(
            api_key="or_test_key",
            model="meta-llama/llama-3.1-8b-instruct:free",
            base_url="https://openrouter.ai/api/v1",
        )
        assert provider.provider_name == "openrouter"
        assert provider.model == "meta-llama/llama-3.1-8b-instruct:free"
        assert provider.base_url == "https://openrouter.ai/api/v1"

    def test_openrouter_provider_default_base_url(self):
        """Test OpenRouterProvider uses default base URL when not provided."""
        provider = OpenRouterProvider(api_key="or_test_key", model="test-model")
        assert provider.base_url == "https://openrouter.ai/api/v1"

    @patch("app.ai.providers.openrouter_provider.OpenAI")
    def test_openrouter_provider_generate_response_success(self, mock_openai_class):
        """Test OpenRouterProvider generates response successfully."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_choice = MagicMock()
        mock_choice.message.content = "OpenRouter response"
        mock_choice.finish_reason = "stop"

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 15
        mock_usage.completion_tokens = 25

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.model = "meta-llama/llama-3.1-8b-instruct:free"
        mock_response.usage = mock_usage

        mock_client.chat.completions.create.return_value = mock_response

        provider = OpenRouterProvider(
            api_key="or_test_key",
            model="meta-llama/llama-3.1-8b-instruct:free",
        )
        messages = [LLMMessage(role="user", content="Test")]
        response = provider.generate_response(messages)

        assert response.content == "OpenRouter response"
        assert response.model == "meta-llama/llama-3.1-8b-instruct:free"
        assert response.usage["prompt_tokens"] == 15
        assert response.usage["completion_tokens"] == 25

        # Verify custom headers were passed
        mock_openai_class.assert_called_once()
        call_kwargs = mock_openai_class.call_args.kwargs
        assert "default_headers" in call_kwargs
        assert call_kwargs["default_headers"]["HTTP-Referer"] == "https://meditrust.local"
        assert call_kwargs["default_headers"]["X-Title"] == "MediTrust AI"


class TestProviderIntegration:
    """Integration tests for provider selection in RAG chain."""

    @patch("app.ai.providers.groq_provider.OpenAI")
    def test_rag_chain_uses_groq_provider(self, mock_openai_class):
        """Test RAG chain uses Groq provider when configured."""
        from app.ai.rag.chain import RAGChain

        # Setup mock
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_choice = MagicMock()
        mock_choice.message.content = "RAG response with Groq"
        mock_choice.finish_reason = "stop"

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 20

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.model = "openai/gpt-oss-120b"
        mock_response.usage = mock_usage

        mock_client.chat.completions.create.return_value = mock_response

        # Configure for Groq
        with _patch_settings(
            LLM_PROVIDER="groq",
            LLM_MODEL="openai/gpt-oss-120b",
            LLM_API_KEY="gsk_test_key",
            LLM_BASE_URL="https://api.groq.com/openai/v1",
        ):
            ProviderFactory.reset()

            chain = RAGChain()
            assert chain.llm_provider.provider_name == "groq"
            assert chain.llm_provider.model == "openai/gpt-oss-120b"

    @patch("app.ai.providers.gemini_provider.genai.Client")
    def test_rag_chain_uses_gemini_provider(self, mock_genai_client):
        """Test RAG chain uses Gemini provider when configured."""
        from app.ai.rag.chain import RAGChain

        # Setup mock
        mock_client_instance = MagicMock()
        mock_genai_client.return_value = mock_client_instance

        mock_usage_metadata = MagicMock()
        mock_usage_metadata.prompt_token_count = 10
        mock_usage_metadata.candidates_token_count = 20

        mock_candidate = MagicMock()
        mock_candidate.finish_reason = "STOP"

        mock_response = MagicMock()
        mock_response.text = "RAG response with Gemini"
        mock_response.usage_metadata = mock_usage_metadata
        mock_response.candidates = [mock_candidate]

        mock_client_instance.models.generate_content.return_value = mock_response

        # Configure for Gemini
        with _patch_settings(
            LLM_PROVIDER="gemini",
            LLM_MODEL="gemini-3.8-flash",
            LLM_API_KEY="gemini_test_key",
            LLM_BASE_URL="",
        ):
            ProviderFactory.reset()

            chain = RAGChain()
            assert chain.llm_provider.provider_name == "gemini"
            assert chain.llm_provider.model == "gemini-3.8-flash"

    def test_groq_provider_does_not_initialize_google_genai(self):
        """Test that selecting Groq does not initialize Google GenAI client."""
        import app.ai.providers.groq_provider as groq_module
        import app.ai.providers.gemini_provider as gemini_module

        # Verify Groq provider imports OpenAI, not google.genai
        assert "openai" in groq_module.__file__ or "OpenAI" in dir(groq_module)

        # The GroqProvider class should not reference google.genai
        import inspect
        groq_source = inspect.getsource(groq_module.GroqProvider)
        assert "google.genai" not in groq_source
        assert "genai" not in groq_source

    def test_gemini_provider_does_not_initialize_openai(self):
        """Test that selecting Gemini does not initialize OpenAI client."""
        import app.ai.providers.gemini_provider as gemini_module

        import inspect
        gemini_source = inspect.getsource(gemini_module.GeminiProvider)
        assert "openai" not in gemini_source.lower()
        assert "OpenAI" not in gemini_source


class TestResponseFormat:
    """Tests for natural, patient-friendly response format without robotic headings."""

    # Headings that should NEVER appear in responses (excluding disclaimer text)
    FORBIDDEN_HEADINGS = [
        "Direct Answer",
        "Explanation",
        "Common Symptoms",
        "Risk Factors",
        "Possible Causes",
        "When to Consult a Doctor",
        "Recommended Doctor",
        "AI Response",
        "Answer:",
        "Response:",
    ]

    @patch("app.ai.providers.groq_provider.OpenAI")
    def test_english_medical_response_no_forbidden_headings(self, mock_openai_class):
        """Test English medical response doesn't contain forbidden headings."""
        from app.ai.rag.chain import RAGChain

        # Setup mock to return a natural medical response
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_choice = MagicMock()
        mock_choice.message.content = (
            "High blood pressure, or hypertension, is a common condition where the force "
            "of blood against your artery walls is consistently too high. It often has no "
            "noticeable symptoms, which is why it's sometimes called a silent condition. "
            "When symptoms do occur, they may include headaches, shortness of breath, or "
            "nosebleeds. You should have your blood pressure checked regularly, especially "
            "if you have a family history of heart disease. If your readings are consistently "
            "above 140/90 mmHg, it's important to consult a doctor for proper evaluation."
        )
        mock_choice.finish_reason = "stop"

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 20

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.model = "openai/gpt-oss-120b"
        mock_response.usage = mock_usage

        mock_client.chat.completions.create.return_value = mock_response

        with _patch_settings(
            LLM_PROVIDER="groq",
            LLM_MODEL="openai/gpt-oss-120b",
            LLM_API_KEY="gsk_test_key",
            LLM_BASE_URL="https://api.groq.com/openai/v1",
        ):
            ProviderFactory.reset()
            chain = RAGChain()

            import asyncio
            response = asyncio.run(chain.generate("What is high blood pressure?", db=None))

            answer = response.answer

            # Check no forbidden headings appear
            for heading in self.FORBIDDEN_HEADINGS:
                assert heading.lower() not in answer.lower(), (
                    f"Forbidden heading '{heading}' found in response"
                )

            # Check response is natural and informative
            assert "hypertension" in answer.lower() or "blood pressure" in answer.lower()
            assert len(answer) > 50  # Not just a one-liner

    @patch("app.ai.providers.groq_provider.OpenAI")
    def test_gujarati_medical_response_no_forbidden_headings(self, mock_openai_class):
        """Test Gujarati medical response doesn't contain forbidden headings."""
        from app.ai.rag.chain import RAGChain

        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_choice = MagicMock()
        mock_choice.message.content = (
            "હાઈ બ્લડ પ્રેશર એવી સ્થિતિ છે જ્યાં લોહીનો દબાવ નિયમિત રીતે વધુ રહે છે. "
            "અને તે ઘણી વખત કોઈ લક્ષણો બતાવતું નથી. જ્યારે લક્ષણો આવે ત્યારે સરદર્દ, "
            "શ્વાસ લેવામાં તકલીફ અથવા નાકમાંથી લોહી આવી શકે છે. નિયમિત તપાસ કરાવવી જોઈએ. "
            "જો તમારા પૅઠિંગ સતત 140/90 mmHg થી વધુ હોય, તો ડૉક્ટરને જોવા જોઈએ."
        )
        mock_choice.finish_reason = "stop"

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 20

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.model = "openai/gpt-oss-120b"
        mock_response.usage = mock_usage

        mock_client.chat.completions.create.return_value = mock_response

        with _patch_settings(
            LLM_PROVIDER="groq",
            LLM_MODEL="openai/gpt-oss-120b",
            LLM_API_KEY="gsk_test_key",
            LLM_BASE_URL="https://api.groq.com/openai/v1",
        ):
            ProviderFactory.reset()
            chain = RAGChain()

            import asyncio
            response = asyncio.run(chain.generate("હાઈ બ્લડ પ્રેશર શું છે?", db=None))

            answer = response.answer

            # Check no forbidden headings appear
            for heading in self.FORBIDDEN_HEADINGS:
                assert heading.lower() not in answer.lower(), (
                    f"Forbidden heading '{heading}' found in Gujarati response"
                )

            # Check response is in Gujarati
            gujarati_chars = sum(1 for c in answer if '\u0A80' <= c <= '\u0AFF')
            assert gujarati_chars > 10, "Response should contain Gujarati characters"

    @patch("app.ai.providers.groq_provider.OpenAI")
    def test_simple_factual_response_is_concise(self, mock_openai_class):
        """Test simple factual questions get brief responses."""
        from app.ai.rag.chain import RAGChain

        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_choice = MagicMock()
        mock_choice.message.content = "The hospital's visiting hours are from 9 AM to 6 PM daily."
        mock_choice.finish_reason = "stop"

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 5
        mock_usage.completion_tokens = 10

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.model = "openai/gpt-oss-120b"
        mock_response.usage = mock_usage

        mock_client.chat.completions.create.return_value = mock_response

        with _patch_settings(
            LLM_PROVIDER="groq",
            LLM_MODEL="openai/gpt-oss-120b",
            LLM_API_KEY="gsk_test_key",
            LLM_BASE_URL="https://api.groq.com/openai/v1",
        ):
            ProviderFactory.reset()
            chain = RAGChain()

            import asyncio
            response = asyncio.run(chain.generate("What are the visiting hours?", db=None))

            answer = response.answer

            # Should be concise (1-3 sentences)
            sentences = answer.count('.') + answer.count('!') + answer.count('?')
            assert sentences <= 3, f"Simple factual response should be brief, got {sentences} sentences"

            # No forbidden headings
            for heading in self.FORBIDDEN_HEADINGS:
                assert heading.lower() not in answer.lower()

    def test_system_prompt_forbids_internal_details(self):
        """Test that system prompt explicitly forbids mentioning internal system details."""
        from app.ai.rag.prompts import build_rag_prompt
        from app.ai.vectorstore.base import SearchResult, VectorDocument

        # Create a mock search result
        doc = VectorDocument(
            id="test_doc",
            content="Hypertension is high blood pressure.",
            embedding=[0.1] * 384,
            metadata={"source": "test.pdf", "page": 1}
        )
        result = SearchResult(document=doc, score=0.9)

        system_prompt, _ = build_rag_prompt("What is hypertension?", [result])

        # Check prompt explicitly forbids internal details
        # These terms are explicitly listed in the prompt
        forbidden_terms = [
            "knowledge base",
            "vector database",
            "rag context",
            "retrieved chunks",
            "source documents",
            "uploaded documents",
            "filenames",
            "source numbers",
        ]

        for term in forbidden_terms:
            assert term.lower() in system_prompt.lower(), (
                f"Prompt should mention '{term}' as forbidden"
            )

        # Check prompt forbids headings
        forbidden_headings = [
            "Direct Answer",
            "Explanation",
            "Common Symptoms",
            "When to Consult a Doctor",
            "Recommended Doctor",
        ]

        for heading in forbidden_headings:
            assert heading in system_prompt, (
                f"Prompt should mention '{heading}' as forbidden heading"
            )

    @patch("app.ai.providers.groq_provider.OpenAI")
    def test_english_question_gets_english_response(self, mock_openai_class):
        """Test English questions get English responses."""
        from app.ai.rag.chain import RAGChain

        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_choice = MagicMock()
        mock_choice.message.content = "Hypertension is high blood pressure. It often has no symptoms."
        mock_choice.finish_reason = "stop"

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 15

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.model = "openai/gpt-oss-120b"
        mock_response.usage = mock_usage

        mock_client.chat.completions.create.return_value = mock_response

        with _patch_settings(
            LLM_PROVIDER="groq",
            LLM_MODEL="openai/gpt-oss-120b",
            LLM_API_KEY="gsk_test_key",
            LLM_BASE_URL="https://api.groq.com/openai/v1",
        ):
            ProviderFactory.reset()
            chain = RAGChain()

            import asyncio
            response = asyncio.run(chain.generate("What is hypertension?", db=None))

            answer = response.answer

            # Should be primarily English (ASCII letters)
            ascii_chars = sum(1 for c in answer if ord(c) < 128)
            gujarati_chars = sum(1 for c in answer if '\u0A80' <= c <= '\u0AFF')
            assert ascii_chars > gujarati_chars * 10, "English question should get English response"

    @patch("app.ai.providers.groq_provider.OpenAI")
    def test_gujarati_question_gets_gujarati_response(self, mock_openai_class):
        """Test Gujarati questions get Gujarati responses."""
        from app.ai.rag.chain import RAGChain

        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_choice = MagicMock()
        mock_choice.message.content = "હાઈપરટેन्शન એ ઉચ્ચ રક્તચાપ છે. ઘણી વખત તેના કોઈ લક્ષણો નથી."
        mock_choice.finish_reason = "stop"

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 15

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.model = "openai/gpt-oss-120b"
        mock_response.usage = mock_usage

        mock_client.chat.completions.create.return_value = mock_response

        with _patch_settings(
            LLM_PROVIDER="groq",
            LLM_MODEL="openai/gpt-oss-120b",
            LLM_API_KEY="gsk_test_key",
            LLM_BASE_URL="https://api.groq.com/openai/v1",
        ):
            ProviderFactory.reset()
            chain = RAGChain()

            import asyncio
            response = asyncio.run(chain.generate("હાઈપરટેन्शન શું છે?", db=None))

            answer = response.answer

            # Should contain Gujarati characters
            gujarati_chars = sum(1 for c in answer if '\u0A80' <= c <= '\u0AFF')
            assert gujarati_chars > 10, "Gujarati question should get Gujarati response"

    def test_prompt_forbids_robotic_headings(self):
        """Test that system prompt explicitly forbids robotic headings."""
        from app.ai.rag.prompts import build_rag_prompt
        from app.ai.vectorstore.base import SearchResult, VectorDocument

        doc = VectorDocument(
            id="test_doc",
            content="Hypertension is high blood pressure.",
            embedding=[0.1] * 384,
            metadata={"source": "test.pdf", "page": 1}
        )
        result = SearchResult(document=doc, score=0.9)

        system_prompt, _ = build_rag_prompt("What is hypertension?", [result])

        # Check prompt explicitly forbids these specific headings
        explicitly_forbidden = [
            "Direct Answer",
            "Explanation",
            "Common Symptoms",
            "When to Consult a Doctor",
            "Recommended Doctor",
        ]

        for heading in explicitly_forbidden:
            assert heading in system_prompt, (
                f"Prompt should mention '{heading}' as forbidden heading"
            )

        # Check prompt says NOT to use headings and covers similar labels
        assert "Do NOT use headings" in system_prompt
        assert "do not use headings" in system_prompt.lower()
        assert "or similar structured labels" in system_prompt.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
