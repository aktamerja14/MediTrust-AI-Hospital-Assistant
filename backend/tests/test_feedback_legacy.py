"""Tests for message feedback with legacy data (messages without IDs)."""
import pytest
from datetime import timedelta, datetime, timezone
from bson import ObjectId

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


class TestMessageFeedbackLegacy:
    """Tests for message feedback with legacy conversations (messages without IDs)."""

    @pytest.mark.asyncio
    async def test_feedback_on_conversation_without_message_ids(self, client, test_db):
        """Test feedback on a conversation where messages lack 'id' fields (legacy data).
        
        This simulates the intermittent failure scenario where:
        1. Conversation was created before the fix (messages have no 'id' field)
        2. User loads conversation -> get_conversation generates IDs but didn't persist them (old behavior)
        3. User submits feedback with generated ID -> backend can't find it in raw MongoDB data
        4. With the fix, get_conversation now persists generated IDs, so feedback works.
        """
        auth_service = AuthService(test_db)
        user_data = UserCreate(email="legacy_feedback@test.com", password="TestPass123")
        user = await auth_service.create_user(user_data, role="patient")
        token = create_token(user.id, "patient")
        headers = {"Authorization": f"Bearer {token}"}

        # Manually insert a conversation WITHOUT message IDs (simulating legacy data)
        now = datetime.now(timezone.utc)
        legacy_conversation = {
            "user_id": user.id,
            "title": "Legacy Conversation",
            "messages": [
                {
                    "role": "user",
                    "content": "Legacy user message",
                    "timestamp": now
                    # No 'id' field!
                },
                {
                    "role": "assistant",
                    "content": "Legacy assistant response",
                    "timestamp": now
                    # No 'id' field!
                }
            ],
            "created_at": now,
            "updated_at": now,
            "last_message_at": now,
        }
        result = await test_db["chat_history"].insert_one(legacy_conversation)
        conv_id = str(result.inserted_id)

        # Verify messages don't have IDs in database
        doc = await test_db["chat_history"].find_one({"_id": result.inserted_id})
        for msg in doc["messages"]:
            assert "id" not in msg, "Legacy messages should not have IDs"

        # Load conversation via API (this should generate and persist IDs with the fix)
        load_resp = await client.get(f"/api/v1/chat/conversations/{conv_id}", headers=headers)
        assert load_resp.status_code == 200
        load_data = load_resp.json()
        
        # Messages should now have IDs
        assert len(load_data["messages"]) == 2
        user_msg_id = load_data["messages"][0]["id"]
        assistant_msg_id = load_data["messages"][1]["id"]
        assert user_msg_id is not None
        assert assistant_msg_id is not None
        assert load_data["messages"][0]["role"] == "user"
        assert load_data["messages"][1]["role"] == "assistant"

        # Verify IDs were persisted to database
        doc_after_load = await test_db["chat_history"].find_one({"_id": result.inserted_id})
        for msg in doc_after_load["messages"]:
            assert "id" in msg, "Messages should have IDs persisted after load"
            assert msg["id"] is not None

        # Now submit feedback on the assistant message - this should work!
        feedback_resp = await client.post("/api/v1/chat/feedback", json={
            "conversation_id": conv_id,
            "message_id": assistant_msg_id,
            "feedback_type": "positive"
        }, headers=headers)
        assert feedback_resp.status_code == 200, f"Feedback failed: {feedback_resp.json()}"
        assert feedback_resp.json()["feedback_type"] == "positive"

        # Submit feedback again (change to negative)
        feedback_resp2 = await client.post("/api/v1/chat/feedback", json={
            "conversation_id": conv_id,
            "message_id": assistant_msg_id,
            "feedback_type": "negative"
        }, headers=headers)
        assert feedback_resp2.status_code == 200
        assert feedback_resp2.json()["feedback_type"] == "negative"

    @pytest.mark.asyncio
    async def test_feedback_on_legacy_conversation_after_new_message(self, client, test_db):
        """Test feedback on legacy conversation after sending a new message.
        
        This tests the scenario where:
        1. Legacy conversation exists (no message IDs)
        2. User sends new message -> send_message generates IDs for old messages and persists them
        3. User submits feedback on old message -> should work because IDs are now persisted
        """
        auth_service = AuthService(test_db)
        user_data = UserCreate(email="legacy_feedback2@test.com", password="TestPass123")
        user = await auth_service.create_user(user_data, role="patient")
        token = create_token(user.id, "patient")
        headers = {"Authorization": f"Bearer {token}"}

        # Manually insert a legacy conversation
        now = datetime.now(timezone.utc)
        legacy_conversation = {
            "user_id": user.id,
            "title": "Legacy Conversation 2",
            "messages": [
                {
                    "role": "user",
                    "content": "Old user message",
                    "timestamp": now
                },
                {
                    "role": "assistant",
                    "content": "Old assistant response",
                    "timestamp": now
                }
            ],
            "created_at": now,
            "updated_at": now,
            "last_message_at": now,
        }
        result = await test_db["chat_history"].insert_one(legacy_conversation)
        conv_id = str(result.inserted_id)

        # Send a new message in this legacy conversation
        resp = await client.post("/api/v1/chat/message", json={
            "message": "New message in legacy conversation",
            "conversation_id": conv_id
        }, headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        
        # The response should contain the new assistant message ID
        new_assistant_msg_id = data["assistant_message"]["id"]
        
        # Load conversation to get the old assistant message ID
        load_resp = await client.get(f"/api/v1/chat/conversations/{conv_id}", headers=headers)
        assert load_resp.status_code == 200
        load_data = load_resp.json()
        
        # Should have 4 messages: old_user, old_assistant, new_user, new_assistant
        assert len(load_data["messages"]) == 4
        old_assistant_msg_id = load_data["messages"][1]["id"]
        
        # Submit feedback on OLD assistant message - should work now!
        feedback_resp = await client.post("/api/v1/chat/feedback", json={
            "conversation_id": conv_id,
            "message_id": old_assistant_msg_id,
            "feedback_type": "positive"
        }, headers=headers)
        assert feedback_resp.status_code == 200, f"Feedback on old message failed: {feedback_resp.json()}"
        assert feedback_resp.json()["feedback_type"] == "positive"

        # Submit feedback on NEW assistant message - should also work
        feedback_resp2 = await client.post("/api/v1/chat/feedback", json={
            "conversation_id": conv_id,
            "message_id": new_assistant_msg_id,
            "feedback_type": "negative"
        }, headers=headers)
        assert feedback_resp2.status_code == 200
        assert feedback_resp2.json()["feedback_type"] == "negative"