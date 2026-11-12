"""Mock LLM Provider for Testing.

Returns canned responses without requiring an API key or external service.
"""
import logging
from typing import List, Optional

from app.ai.providers.base import (
    BaseLLMProvider,
    LLMMessage,
    LLMResponse,
)

logger = logging.getLogger("meditrust.ai.providers.mock")


class MockProvider(BaseLLMProvider):
    """Mock LLM provider for testing and development."""

    def __init__(
        self,
        api_key: str = "mock-key",
        model: str = "mock-model",
        temperature: float = 0.2,
        max_tokens: int = 2048,
        timeout: int = 60,
        base_url: str = "",
    ):
        super().__init__(api_key, model, temperature, max_tokens, timeout, base_url)

    def _validate_config(self) -> None:
        """Mock provider doesn't require API key validation."""
        pass

    @property
    def provider_name(self) -> str:
        return "mock"

    def generate_response(
        self,
        messages: List[LLMMessage],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Generate a canned mock response."""
        user_message = ""
        for msg in messages:
            if msg.role == "user":
                user_message = msg.content
                break

        # Simple canned responses based on keywords
        if "blood pressure" in user_message.lower() or "hypertension" in user_message.lower():
            content = "High blood pressure (hypertension) often has no symptoms. When symptoms occur, they may include headaches, shortness of breath, nosebleeds, or dizziness. Regular monitoring is important."
        elif "upload" in user_message.lower() or "document" in user_message.lower():
            content = "To upload a document, use the admin dashboard. Go to the Documents section and click Upload. Supported formats: PDF and TXT, max 10MB."
        elif "hello" in user_message.lower() or "hi" in user_message.lower():
            content = "Hello! I'm the MediTrust AI assistant. How can I help you today?"
        elif "symptom" in user_message.lower():
            content = "Common symptoms vary by condition. Please consult a healthcare professional for accurate diagnosis."
        else:
            content = f"Mock response to: {user_message[:100]}"

        return LLMResponse(
            content=content,
            model=self.model,
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            finish_reason="stop",
        )