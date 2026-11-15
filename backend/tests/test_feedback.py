"""Tests for message feedback functionality."""
import pytest
from datetime import timedelta

from app.main import create_application
from app.database.connection import DatabaseManager
from app.services.auth import AuthService
from app.schemas.auth import UserCreate
from app.core.security.jwt import create_access_token
from httpx import AsyncClient, ASGITransport
from tests.conftest import TEST_DATABASE_NAME, verify_test_database


def create_token(user_id: str, role: str, expires_delta: timedelta = timedelta(minutes=30)) -> str:
    """Create a JWT token for testing."""
    return create_access_token(subject=user_id, role=role, expires_delta=expires_delta)


class TestMessageFeedback:
    """Tests for message feedback endpoints."""

    @pytest.mark.asyncio
    async def test_submit_feedback_on_assistant_message(self, client, test_db):
        """Test submitting positive feedback on an assistant message."""
        auth_service = AuthService(test_db)
        user_data = UserCreate(email="feedback_test@test.com", password="TestPass123")
        user = await auth_service.create_user(user_data, role="patient")
        token = create_token(user.id, "patient")
        headers = {"Authorization": f"Bearer {token}"}

        # Send a message
        resp = await client.post("/api/v1/chat/message", json={
            "message": "Test message for feedback"
        }, headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        conv_id = data["conversation_id"]
        assistant_msg_id = data["assistant_message"]["id"]

        # Submit positive feedback
        feedback_resp = await client.post("/api/v1/chat/feedback", json={
            "conversation_id": conv_id,
            "message_id": assistant_msg_id,
            "feedback_type": "positive"
        }, headers=headers)
        assert feedback_resp.status_code == 200
        feedback_data = feedback_resp.json()
        assert feedback_data["feedback_type"] == "positive"
        assert feedback_data["message_id"] == assistant_msg_id
        assert feedback_data["conversation_id"] == conv_id

    @pytest.mark.asyncio
    async def test_submit_negative_feedback(self, client, test_db):
        """Test submitting negative feedback on an assistant message."""
        auth_service = AuthService(test_db)
        user_data = UserCreate(email="feedback_test2@test.com", password="TestPass123")
        user = await auth_service.create_user(user_data, role="patient")
        token = create_token(user.id, "patient")
        headers = {"Authorization": f"Bearer {token}"}

        # Send a message
        resp = await client.post("/api/v1/chat/message", json={
            "message": "Test message for negative feedback"
        }, headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        conv_id = data["conversation_id"]
        assistant_msg_id = data["assistant_message"]["id"]

        # Submit negative feedback
        feedback_resp = await client.post("/api/v1/chat/feedback", json={
            "conversation_id": conv_id,
            "message_id": assistant_msg_id,
            "feedback_type": "negative"
        }, headers=headers)
        assert feedback_resp.status_code == 200
        feedback_data = feedback_resp.json()
        assert feedback_data["feedback_type"] == "negative"

    @pytest.mark.asyncio
    async def test_change_feedback_from_positive_to_negative(self, client, test_db):
        """Test changing feedback from positive to negative."""
        auth_service = AuthService(test_db)
        user_data = UserCreate(email="feedback_test3@test.com", password="TestPass123")
        user = await auth_service.create_user(user_data, role="patient")
        token = create_token(user.id, "patient")
        headers = {"Authorization": f"Bearer {token}"}

        # Send a message
        resp = await client.post("/api/v1/chat/message", json={
            "message": "Test message for changing feedback"
        }, headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        conv_id = data["conversation_id"]
        assistant_msg_id = data["assistant_message"]["id"]

        # Submit positive feedback
        feedback_resp = await client.post("/api/v1/chat/feedback", json={
            "conversation_id": conv_id,
            "message_id": assistant_msg_id,
            "feedback_type": "positive"
        }, headers=headers)
        assert feedback_resp.status_code == 200
        assert feedback_resp.json()["feedback_type"] == "positive"

        # Change to negative feedback
        feedback_resp2 = await client.post("/api/v1/chat/feedback", json={
            "conversation_id": conv_id,
            "message_id": assistant_msg_id,
            "feedback_type": "negative"
        }, headers=headers)
        assert feedback_resp2.status_code == 200
        assert feedback_resp2.json()["feedback_type"] == "negative"

    @pytest.mark.asyncio
    async def test_feedback_rejected_on_user_message(self, client, test_db):
        """Test that feedback on user messages is rejected."""
        auth_service = AuthService(test_db)
        user_data = UserCreate(email="feedback_test4@test.com", password="TestPass123")
        user = await auth_service.create_user(user_data, role="patient")
        token = create_token(user.id, "patient")
        headers = {"Authorization": f"Bearer {token}"}

        # Send a message
        resp = await client.post("/api/v1/chat/message", json={
            "message": "Test message for user feedback rejection"
        }, headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        conv_id = data["conversation_id"]
        user_msg_id = data["user_message"]["id"]

        # Try to submit feedback on user message
        feedback_resp = await client.post("/api/v1/chat/feedback", json={
            "conversation_id": conv_id,
            "message_id": user_msg_id,
            "feedback_type": "positive"
        }, headers=headers)
        assert feedback_resp.status_code == 404
        assert "not an assistant message" in feedback_resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_feedback_rejected_on_invalid_message_id(self, client, test_db):
        """Test that feedback with invalid message ID is rejected."""
        auth_service = AuthService(test_db)
        user_data = UserCreate(email="feedback_test5@test.com", password="TestPass123")
        user = await auth_service.create_user(user_data, role="patient")
        token = create_token(user.id, "patient")
        headers = {"Authorization": f"Bearer {token}"}

        # Send a message
        resp = await client.post("/api/v1/chat/message", json={
            "message": "Test message for invalid ID"
        }, headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        conv_id = data["conversation_id"]

        # Try to submit feedback with invalid message ID
        feedback_resp = await client.post("/api/v1/chat/feedback", json={
            "conversation_id": conv_id,
            "message_id": "invalid_id_that_does_not_exist",
            "feedback_type": "positive"
        }, headers=headers)
        assert feedback_resp.status_code == 404
        assert "not an assistant message" in feedback_resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_feedback_after_reload_conversation(self, client, test_db):
        """Test that feedback works after reloading conversation (message IDs persist)."""
        auth_service = AuthService(test_db)
        user_data = UserCreate(email="feedback_test6@test.com", password="TestPass123")
        user = await auth_service.create_user(user_data, role="patient")
        token = create_token(user.id, "patient")
        headers = {"Authorization": f"Bearer {token}"}

        # Send a message
        resp = await client.post("/api/v1/chat/message", json={
            "message": "Test message for reload feedback"
        }, headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        conv_id = data["conversation_id"]
        assistant_msg_id = data["assistant_message"]["id"]

        # Reload conversation
        load_resp = await client.get(f"/api/v1/chat/conversations/{conv_id}", headers=headers)
        assert load_resp.status_code == 200
        load_data = load_resp.json()
        loaded_assistant_msg_id = load_data["messages"][1]["id"]

        # Message ID should match after reload
        assert loaded_assistant_msg_id == assistant_msg_id, "Message ID mismatch after reload!"

        # Submit feedback on loaded message
        feedback_resp = await client.post("/api/v1/chat/feedback", json={
            "conversation_id": conv_id,
            "message_id": loaded_assistant_msg_id,
            "feedback_type": "positive"
        }, headers=headers)
        assert feedback_resp.status_code == 200
        assert feedback_resp.json()["feedback_type"] == "positive"

    @pytest.mark.asyncio
    async def test_feedback_on_old_conversation(self, client, test_db):
        """Test feedback on messages from an older conversation (simulating pre-fix data)."""
        auth_service = AuthService(test_db)
        user_data = UserCreate(email="feedback_test7@test.com", password="TestPass123")
        user = await auth_service.create_user(user_data, role="patient")
        token = create_token(user.id, "patient")
        headers = {"Authorization": f"Bearer {token}"}

        # Send a message to create conversation
        resp = await client.post("/api/v1/chat/message", json={
            "message": "First message"
        }, headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        conv_id = data["conversation_id"]

        # Send another message in same conversation
        resp2 = await client.post("/api/v1/chat/message", json={
            "message": "Second message"
        }, headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
        # Note: client.post doesn't merge headers, need to pass full headers
        resp2 = await client.post("/api/v1/chat/message", json={
            "message": "Second message",
            "conversation_id": conv_id
        }, headers=headers)
        assert resp2.status_code == 200
        data2 = resp2.json()
        assistant_msg_id_2 = data2["assistant_message"]["id"]

        # Reload conversation (simulates user coming back later)
        load_resp = await client.get(f"/api/v1/chat/conversations/{conv_id}", headers=headers)
        assert load_resp.status_code == 200
        load_data = load_resp.json()
        # Should have 4 messages: user1, assistant1, user2, assistant2
        assert len(load_data["messages"]) == 4
        loaded_assistant_msg_id_2 = load_data["messages"][3]["id"]
        assert loaded_assistant_msg_id_2 == assistant_msg_id_2

        # Submit feedback on second assistant message
        feedback_resp = await client.post("/api/v1/chat/feedback", json={
            "conversation_id": conv_id,
            "message_id": loaded_assistant_msg_id_2,
            "feedback_type": "positive"
        }, headers=headers)
        assert feedback_resp.status_code == 200
        assert feedback_resp.json()["feedback_type"] == "positive"

    @pytest.mark.asyncio
    async def test_get_message_feedback(self, client, test_db):
        """Test retrieving feedback for a message."""
        auth_service = AuthService(test_db)
        user_data = UserCreate(email="feedback_test8@test.com", password="TestPass123")
        user = await auth_service.create_user(user_data, role="patient")
        token = create_token(user.id, "patient")
        headers = {"Authorization": f"Bearer {token}"}

        # Send a message
        resp = await client.post("/api/v1/chat/message", json={
            "message": "Test message for get feedback"
        }, headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        conv_id = data["conversation_id"]
        assistant_msg_id = data["assistant_message"]["id"]

        # Get feedback (should be null initially)
        get_resp = await client.get(f"/api/v1/chat/feedback/{assistant_msg_id}?conversation_id={conv_id}", headers=headers)
        assert get_resp.status_code == 200
        assert get_resp.json() is None

        # Submit feedback
        feedback_resp = await client.post("/api/v1/chat/feedback", json={
            "conversation_id": conv_id,
            "message_id": assistant_msg_id,
            "feedback_type": "negative"
        }, headers=headers)
        assert feedback_resp.status_code == 200

        # Get feedback again
        get_resp2 = await client.get(f"/api/v1/chat/feedback/{assistant_msg_id}?conversation_id={conv_id}", headers=headers)
        assert get_resp2.status_code == 200
        feedback_data = get_resp2.json()
        assert feedback_data is not None
        assert feedback_data["feedback_type"] == "negative"
        assert feedback_data["message_id"] == assistant_msg_id