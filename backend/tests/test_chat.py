"""Tests for AI Chatbot API endpoints - accessible by both patients and admins."""
import pytest
from datetime import timedelta, datetime, timezone
import re

from app.main import create_application
from app.database.connection import DatabaseManager
from app.services.auth import AuthService
from app.schemas.auth import UserCreate
from app.core.security.jwt import create_access_token
from tests.conftest import TEST_DATABASE_NAME, verify_test_database


def create_token(user_id: str, role: str, expires_delta: timedelta = timedelta(minutes=30)) -> str:
    """Create a JWT token for testing."""
    return create_access_token(subject=user_id, role=role, expires_delta=expires_delta)


def assert_iso8601_utc(timestamp_str: str):
    """Assert timestamp is ISO 8601 format with UTC timezone."""
    # Pattern for ISO 8601 with timezone (Z or +00:00 or -00:00)
    pattern = r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:?\d{2})$'
    assert re.match(pattern, timestamp_str), f"Timestamp '{timestamp_str}' is not valid ISO 8601 with timezone"
    # Parse and verify it's UTC
    dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
    assert dt.tzinfo is not None, f"Timestamp '{timestamp_str}' is not timezone-aware"
    # Verify it's UTC (offset 0)
    assert dt.utcoffset() == timedelta(0), f"Timestamp '{timestamp_str}' is not UTC"


class TestChatEndpoints:
    """Tests for AI Chatbot API endpoints."""

    @pytest.mark.asyncio
    async def test_patient_can_send_message(self, client, test_db):
        """Test patient can send a message to the chatbot."""
        auth_service = AuthService(test_db)
        user_data = UserCreate(email="patient@test.com", password="patientPass123")
        patient_user = await auth_service.create_user(user_data, role="patient")

        token = create_token(patient_user.id, "patient")
        headers = {"Authorization": f"Bearer {token}"}

        response = await client.post("/api/v1/chat/message", json={
            "message": "What are the symptoms of high blood pressure?"
        }, headers=headers)

        assert response.status_code == 200
        data = response.json()
        assert "conversation_id" in data
        assert "user_message" in data
        assert "assistant_message" in data
        assert data["user_message"]["role"] == "user"
        assert data["user_message"]["content"] == "What are the symptoms of high blood pressure?"
        assert data["assistant_message"]["role"] == "assistant"
        assert len(data["assistant_message"]["content"]) > 0

    @pytest.mark.asyncio
    async def test_admin_can_send_message(self, client, test_db):
        """Test admin can send a message to the chatbot."""
        auth_service = AuthService(test_db)
        user_data = UserCreate(email="admin@test.com", password="adminPass123")
        admin_user = await auth_service.create_user(user_data, role="admin")

        token = create_token(admin_user.id, "admin")
        headers = {"Authorization": f"Bearer {token}"}

        response = await client.post("/api/v1/chat/message", json={
            "message": "How do I upload a document?"
        }, headers=headers)

        assert response.status_code == 200
        data = response.json()
        assert "conversation_id" in data
        assert "user_message" in data
        assert "assistant_message" in data
        assert data["user_message"]["role"] == "user"
        assert data["assistant_message"]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_patient_can_continue_conversation(self, client, test_db):
        """Test patient can continue an existing conversation."""
        auth_service = AuthService(test_db)
        user_data = UserCreate(email="patient2@test.com", password="patientPass123")
        patient_user = await auth_service.create_user(user_data, role="patient")

        token = create_token(patient_user.id, "patient")
        headers = {"Authorization": f"Bearer {token}"}

        # Send first message
        response1 = await client.post("/api/v1/chat/message", json={
            "message": "Hello"
        }, headers=headers)
        assert response1.status_code == 200
        conversation_id = response1.json()["conversation_id"]

        # Send second message with conversation_id
        response2 = await client.post("/api/v1/chat/message", json={
            "message": "How are you?",
            "conversation_id": conversation_id
        }, headers=headers)

        assert response2.status_code == 200
        data = response2.json()
        assert data["conversation_id"] == conversation_id
        # Should have both messages in the conversation
        assert len(data["user_message"]["content"]) > 0

    @pytest.mark.asyncio
    async def test_patient_can_list_conversations(self, client, test_db):
        """Test patient can list their conversations."""
        auth_service = AuthService(test_db)
        user_data = UserCreate(email="patient3@test.com", password="patientPass123")
        patient_user = await auth_service.create_user(user_data, role="patient")

        token = create_token(patient_user.id, "patient")
        headers = {"Authorization": f"Bearer {token}"}

        # Send a message first
        await client.post("/api/v1/chat/message", json={
            "message": "Test message for listing"
        }, headers=headers)

        # List conversations
        response = await client.get("/api/v1/chat/conversations", headers=headers)

        assert response.status_code == 200
        data = response.json()
        assert "conversations" in data
        assert len(data["conversations"]) >= 1
        conv = data["conversations"][0]
        assert "id" in conv
        assert "title" in conv
        assert "preview" in conv
        assert "updated_at" in conv

    @pytest.mark.asyncio
    async def test_admin_can_list_conversations(self, client, test_db):
        """Test admin can list their conversations."""
        auth_service = AuthService(test_db)
        user_data = UserCreate(email="admin2@test.com", password="adminPass123")
        admin_user = await auth_service.create_user(user_data, role="admin")

        token = create_token(admin_user.id, "admin")
        headers = {"Authorization": f"Bearer {token}"}

        # Send a message first
        await client.post("/api/v1/chat/message", json={
            "message": "Admin test message"
        }, headers=headers)

        # List conversations
        response = await client.get("/api/v1/chat/conversations", headers=headers)

        assert response.status_code == 200
        data = response.json()
        assert "conversations" in data
        assert len(data["conversations"]) >= 1

    @pytest.mark.asyncio
    async def test_patient_can_get_conversation(self, client, test_db):
        """Test patient can get a specific conversation with messages."""
        auth_service = AuthService(test_db)
        user_data = UserCreate(email="patient4@test.com", password="patientPass123")
        patient_user = await auth_service.create_user(user_data, role="patient")

        token = create_token(patient_user.id, "patient")
        headers = {"Authorization": f"Bearer {token}"}

        # Send a message
        send_response = await client.post("/api/v1/chat/message", json={
            "message": "Get conversation test"
        }, headers=headers)
        conversation_id = send_response.json()["conversation_id"]

        # Get the conversation
        response = await client.get(f"/api/v1/chat/conversations/{conversation_id}", headers=headers)

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == conversation_id
        assert data["user_id"] == patient_user.id
        assert "messages" in data
        assert len(data["messages"]) == 2  # user + assistant
        assert data["messages"][0]["role"] == "user"
        assert data["messages"][1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_patient_cannot_access_other_user_conversation(self, client, test_db):
        """Test patient cannot access another user's conversation."""
        auth_service = AuthService(test_db)

        # Create two patients
        user1_data = UserCreate(email="patientA@test.com", password="patientPass123")
        patient1 = await auth_service.create_user(user1_data, role="patient")

        user2_data = UserCreate(email="patientB@test.com", password="patientPass123")
        patient2 = await auth_service.create_user(user2_data, role="patient")

        token1 = create_token(patient1.id, "patient")
        headers1 = {"Authorization": f"Bearer {token1}"}

        # Patient 1 creates a conversation
        send_response = await client.post("/api/v1/chat/message", json={
            "message": "Private message"
        }, headers=headers1)
        conversation_id = send_response.json()["conversation_id"]

        # Patient 2 tries to access it
        token2 = create_token(patient2.id, "patient")
        headers2 = {"Authorization": f"Bearer {token2}"}

        response = await client.get(f"/api/v1/chat/conversations/{conversation_id}", headers=headers2)

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_patient_can_delete_conversation(self, client, test_db):
        """Test patient can delete their conversation."""
        auth_service = AuthService(test_db)
        user_data = UserCreate(email="patient5@test.com", password="patientPass123")
        patient_user = await auth_service.create_user(user_data, role="patient")

        token = create_token(patient_user.id, "patient")
        headers = {"Authorization": f"Bearer {token}"}

        # Send a message
        send_response = await client.post("/api/v1/chat/message", json={
            "message": "To be deleted"
        }, headers=headers)
        conversation_id = send_response.json()["conversation_id"]

        # Delete the conversation
        response = await client.delete(f"/api/v1/chat/conversations/{conversation_id}", headers=headers)

        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Conversation deleted successfully"
        assert data["conversation_id"] == conversation_id

        # Verify it's gone
        list_response = await client.get("/api/v1/chat/conversations", headers=headers)
        assert list_response.json()["conversations"] == []

    @pytest.mark.asyncio
    async def test_missing_token_returns_401(self, client):
        """Test chat endpoints return 401 without token."""
        response = await client.post("/api/v1/chat/message", json={
            "message": "Hello"
        })
        assert response.status_code == 401

        response = await client.get("/api/v1/chat/conversations")
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_invalid_token_returns_401(self, client):
        """Test chat endpoints return 401 with invalid token."""
        headers = {"Authorization": "Bearer invalid.token.here"}

        response = await client.post("/api/v1/chat/message", json={
            "message": "Hello"
        }, headers=headers)
        assert response.status_code == 401

        response = await client.get("/api/v1/chat/conversations", headers=headers)
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_validation_error_empty_message(self, client, test_db):
        """Test validation error for empty message."""
        auth_service = AuthService(test_db)
        user_data = UserCreate(email="patient6@test.com", password="patientPass123")
        patient_user = await auth_service.create_user(user_data, role="patient")

        token = create_token(patient_user.id, "patient")
        headers = {"Authorization": f"Bearer {token}"}

        response = await client.post("/api/v1/chat/message", json={
            "message": ""
        }, headers=headers)

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_validation_error_message_too_long(self, client, test_db):
        """Test validation error for message too long."""
        auth_service = AuthService(test_db)
        user_data = UserCreate(email="patient7@test.com", password="patientPass123")
        patient_user = await auth_service.create_user(user_data, role="patient")

        token = create_token(patient_user.id, "patient")
        headers = {"Authorization": f"Bearer {token}"}

        long_message = "x" * 4001  # Exceeds max_length of 4000
        response = await client.post("/api/v1/chat/message", json={
            "message": long_message
        }, headers=headers)

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_timestamps_are_iso8601_utc(self, client, test_db):
        """Test that all timestamps in API responses are ISO 8601 UTC format."""
        auth_service = AuthService(test_db)
        user_data = UserCreate(email="patient_ts@test.com", password="patientPass123")
        patient_user = await auth_service.create_user(user_data, role="patient")

        token = create_token(patient_user.id, "patient")
        headers = {"Authorization": f"Bearer {token}"}

        # Send a message
        response = await client.post("/api/v1/chat/message", json={
            "message": "Test timestamps"
        }, headers=headers)

        assert response.status_code == 200
        data = response.json()

        # Check message timestamps
        assert_iso8601_utc(data["user_message"]["timestamp"])
        assert_iso8601_utc(data["assistant_message"]["timestamp"])

        # List conversations and check timestamps
        response = await client.get("/api/v1/chat/conversations", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert len(data["conversations"]) >= 1
        conv = data["conversations"][0]
        assert_iso8601_utc(conv["last_message_at"])
        assert_iso8601_utc(conv["updated_at"])

        # Get conversation and check timestamps
        response = await client.get(f"/api/v1/chat/conversations/{data['conversations'][0]['id']}", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert_iso8601_utc(data["created_at"])
        assert_iso8601_utc(data["updated_at"])
        assert_iso8601_utc(data["last_message_at"])
        for msg in data["messages"]:
            assert_iso8601_utc(msg["timestamp"])

    @pytest.mark.asyncio
    async def test_patient_can_rename_conversation(self, client, test_db):
        """Test patient can rename their conversation."""
        auth_service = AuthService(test_db)
        user_data = UserCreate(email="patient_rename@test.com", password="patientPass123")
        patient_user = await auth_service.create_user(user_data, role="patient")

        token = create_token(patient_user.id, "patient")
        headers = {"Authorization": f"Bearer {token}"}

        # Send a message to create conversation
        send_response = await client.post("/api/v1/chat/message", json={
            "message": "Conversation to rename"
        }, headers=headers)
        conversation_id = send_response.json()["conversation_id"]

        # Rename the conversation
        new_title = "Renamed Conversation Title"
        response = await client.patch(
            f"/api/v1/chat/conversations/{conversation_id}",
            json={"title": new_title},
            headers=headers
        )

        assert response.status_code == 200
        data = response.json()
        assert data["title"] == new_title
        assert data["id"] == conversation_id
        assert data["user_id"] == patient_user.id

        # Verify the change persists in list
        list_response = await client.get("/api/v1/chat/conversations", headers=headers)
        assert list_response.status_code == 200
        convs = list_response.json()["conversations"]
        assert any(c["title"] == new_title for c in convs)

    @pytest.mark.asyncio
    async def test_patient_cannot_rename_other_user_conversation(self, client, test_db):
        """Test patient cannot rename another user's conversation."""
        auth_service = AuthService(test_db)

        # Create two patients
        user1_data = UserCreate(email="patientR1@test.com", password="patientPass123")
        patient1 = await auth_service.create_user(user1_data, role="patient")

        user2_data = UserCreate(email="patientR2@test.com", password="patientPass123")
        patient2 = await auth_service.create_user(user2_data, role="patient")

        token1 = create_token(patient1.id, "patient")
        headers1 = {"Authorization": f"Bearer {token1}"}

        # Patient 1 creates a conversation
        send_response = await client.post("/api/v1/chat/message", json={
            "message": "Private message"
        }, headers=headers1)
        conversation_id = send_response.json()["conversation_id"]

        # Patient 2 tries to rename it
        token2 = create_token(patient2.id, "patient")
        headers2 = {"Authorization": f"Bearer {token2}"}

        response = await client.patch(
            f"/api/v1/chat/conversations/{conversation_id}",
            json={"title": "Hacked Title"},
            headers=headers2
        )

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_conversations_sorted_by_last_message_at(self, client, test_db):
        """Test conversations are sorted by last_message_at descending."""
        auth_service = AuthService(test_db)
        user_data = UserCreate(email="patient_sort@test.com", password="patientPass123")
        patient_user = await auth_service.create_user(user_data, role="patient")

        token = create_token(patient_user.id, "patient")
        headers = {"Authorization": f"Bearer {token}"}

        # Create first conversation
        await client.post("/api/v1/chat/message", json={"message": "First conversation"}, headers=headers)
        
        # Small delay to ensure different timestamps
        import asyncio
        await asyncio.sleep(0.1)
        
        # Create second conversation
        await client.post("/api/v1/chat/message", json={"message": "Second conversation"}, headers=headers)
        
        await asyncio.sleep(0.1)
        
        # Create third conversation
        await client.post("/api/v1/chat/message", json={"message": "Third conversation"}, headers=headers)

        # List conversations - should be sorted by last_message_at descending (newest first)
        response = await client.get("/api/v1/chat/conversations", headers=headers)
        assert response.status_code == 200
        data = response.json()
        conversations = data["conversations"]
        assert len(conversations) >= 3

        # Verify descending order by last_message_at
        for i in range(len(conversations) - 1):
            current = datetime.fromisoformat(conversations[i]["last_message_at"].replace('Z', '+00:00'))
            next_conv = datetime.fromisoformat(conversations[i + 1]["last_message_at"].replace('Z', '+00:00'))
            assert current >= next_conv, "Conversations not sorted by last_message_at descending"