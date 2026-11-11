"""Tests for AI Chatbot API endpoints - accessible by both patients and admins."""
import pytest
from datetime import timedelta

from app.main import create_application
from app.database.connection import DatabaseManager
from app.services.auth import AuthService
from app.schemas.auth import UserCreate
from app.core.security.jwt import create_access_token
from tests.conftest import TEST_DATABASE_NAME, verify_test_database


def create_token(user_id: str, role: str, expires_delta: timedelta = timedelta(minutes=30)) -> str:
    """Create a JWT token for testing."""
    return create_access_token(subject=user_id, role=role, expires_delta=expires_delta)


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