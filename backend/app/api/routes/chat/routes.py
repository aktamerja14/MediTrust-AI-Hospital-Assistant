"""AI Chatbot API routes - accessible by both patients and admins."""
import logging
from typing import Annotated, Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from pymongo.asynchronous.database import AsyncDatabase

from app.database.connection import get_db
from app.api.dependencies import get_auth_service, get_current_active_user, require_any_role
from app.services.auth import AuthService
from app.schemas.auth import UserResponse

logger = logging.getLogger("meditrust.api.chat")

router = APIRouter(prefix="/chat", tags=["AI Chatbot"])

CHAT_HISTORY_COLLECTION = "chat_history"


class ChatMessage(BaseModel):
    """Chat message structure."""
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Message timestamp")


class ChatConversation(BaseModel):
    """Chat conversation structure."""
    id: str = Field(..., description="Conversation unique identifier")
    user_id: str = Field(..., description="User ID who owns this conversation")
    title: str = Field(..., description="Conversation title")
    messages: list[ChatMessage] = Field(default_factory=list, description="List of messages")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Conversation creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")


class ChatRequest(BaseModel):
    """Request to send a message in a conversation."""
    message: str = Field(..., min_length=1, max_length=4000, description="User message")
    conversation_id: Optional[str] = Field(None, description="Existing conversation ID (optional)")


class ChatResponse(BaseModel):
    """Response from chatbot."""
    conversation_id: str = Field(..., description="Conversation ID")
    user_message: ChatMessage = Field(..., description="Echo of user message")
    assistant_message: ChatMessage = Field(..., description="AI assistant response")


class ConversationListItem(BaseModel):
    """Conversation summary for listing."""
    id: str
    title: str
    preview: str
    updated_at: datetime


class ConversationListResponse(BaseModel):
    """Response for listing conversations."""
    conversations: list[ConversationListItem]


@router.post(
    "/message",
    response_model=ChatResponse,
    summary="Send a message to the AI chatbot",
    responses={
        200: {"description": "Message processed successfully"},
        401: {"description": "Not authenticated"},
        403: {"description": "Inactive user account"},
        422: {"description": "Validation error"},
    },
    dependencies=[Depends(require_any_role)],
)
async def send_message(
    request: ChatRequest,
    current_user: Annotated[UserResponse, Depends(get_current_active_user)],
    db: Annotated[AsyncDatabase, Depends(get_db)],
) -> ChatResponse:
    """
    Send a message to the AI chatbot and receive a response.

    Both patients and admins can use this endpoint.
    Creates a new conversation if conversation_id is not provided.
    """
    # Get or create conversation
    conversation = None
    if request.conversation_id:
        try:
            from bson import ObjectId
            conversation_doc = await db[CHAT_HISTORY_COLLECTION].find_one({
                "_id": ObjectId(request.conversation_id),
                "user_id": current_user.id
            })
            if conversation_doc:
                conversation = ChatConversation(
                    id=str(conversation_doc["_id"]),
                    user_id=conversation_doc["user_id"],
                    title=conversation_doc["title"],
                    messages=[ChatMessage(**msg) for msg in conversation_doc.get("messages", [])],
                    created_at=conversation_doc["created_at"],
                    updated_at=conversation_doc["updated_at"],
                )
        except Exception:
            pass

    if not conversation:
        # Create new conversation
        conversation = ChatConversation(
            id="",  # Will be set after insert
            user_id=current_user.id,
            title=request.message[:50] + ("..." if len(request.message) > 50 else ""),
            messages=[],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

    # Add user message
    user_message = ChatMessage(role="user", content=request.message)
    conversation.messages.append(user_message)

    # Generate AI response (placeholder - replace with actual AI integration)
    assistant_response = generate_ai_response(request.message, current_user.role)
    assistant_message = ChatMessage(role="assistant", content=assistant_response)
    conversation.messages.append(assistant_message)

    # Update timestamp
    conversation.updated_at = datetime.utcnow()

    # Save to database
    conversation_doc = {
        "user_id": conversation.user_id,
        "title": conversation.title,
        "messages": [msg.model_dump() for msg in conversation.messages],
        "created_at": conversation.created_at,
        "updated_at": conversation.updated_at,
    }

    if conversation.id:
        # Update existing
        from bson import ObjectId
        await db[CHAT_HISTORY_COLLECTION].update_one(
            {"_id": ObjectId(conversation.id)},
            {"$set": conversation_doc}
        )
    else:
        # Insert new
        result = await db[CHAT_HISTORY_COLLECTION].insert_one(conversation_doc)
        conversation.id = str(result.inserted_id)

    logger.info("Chat message processed for user: %s (role: %s)", current_user.email, current_user.role)

    return ChatResponse(
        conversation_id=conversation.id,
        user_message=user_message,
        assistant_message=assistant_message,
    )


@router.get(
    "/conversations",
    response_model=ConversationListResponse,
    summary="List user's chat conversations",
    responses={
        200: {"description": "List of conversations"},
        401: {"description": "Not authenticated"},
        403: {"description": "Inactive user account"},
    },
    dependencies=[Depends(require_any_role)],
)
async def list_conversations(
    current_user: Annotated[UserResponse, Depends(get_current_active_user)],
    db: Annotated[AsyncDatabase, Depends(get_db)],
) -> ConversationListResponse:
    """
    List all chat conversations for the current user.

    Both patients and admins can use this endpoint.
    Returns conversations sorted by most recent first.
    """
    cursor = db[CHAT_HISTORY_COLLECTION].find(
        {"user_id": current_user.id},
        {"title": 1, "messages": 1, "updated_at": 1}
    ).sort("updated_at", -1)

    conversations = []
    async for doc in cursor:
        messages = doc.get("messages", [])
        preview = ""
        if messages:
            last_msg = messages[-1]
            preview = last_msg.get("content", "")[:100]
            if len(last_msg.get("content", "")) > 100:
                preview += "..."

        conversations.append(ConversationListItem(
            id=str(doc["_id"]),
            title=doc.get("title", "Untitled"),
            preview=preview,
            updated_at=doc.get("updated_at", doc.get("created_at", datetime.utcnow())),
        ))

    return ConversationListResponse(conversations=conversations)


@router.get(
    "/conversations/{conversation_id}",
    response_model=ChatConversation,
    summary="Get a specific conversation with all messages",
    responses={
        200: {"description": "Conversation with messages"},
        401: {"description": "Not authenticated"},
        403: {"description": "Inactive user account or access denied"},
        404: {"description": "Conversation not found"},
    },
    dependencies=[Depends(require_any_role)],
)
async def get_conversation(
    conversation_id: str,
    current_user: Annotated[UserResponse, Depends(get_current_active_user)],
    db: Annotated[AsyncDatabase, Depends(get_db)],
) -> ChatConversation:
    """
    Get a specific conversation with all messages.

    Both patients and admins can use this endpoint.
    Users can only access their own conversations.
    """
    try:
        from bson import ObjectId
        obj_id = ObjectId(conversation_id)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )

    conversation_doc = await db[CHAT_HISTORY_COLLECTION].find_one({
        "_id": obj_id,
        "user_id": current_user.id
    })

    if not conversation_doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )

    return ChatConversation(
        id=str(conversation_doc["_id"]),
        user_id=conversation_doc["user_id"],
        title=conversation_doc["title"],
        messages=[ChatMessage(**msg) for msg in conversation_doc.get("messages", [])],
        created_at=conversation_doc["created_at"],
        updated_at=conversation_doc["updated_at"],
    )


@router.delete(
    "/conversations/{conversation_id}",
    response_model=dict,
    summary="Delete a conversation",
    responses={
        200: {"description": "Conversation deleted successfully"},
        401: {"description": "Not authenticated"},
        403: {"description": "Inactive user account or access denied"},
        404: {"description": "Conversation not found"},
    },
    dependencies=[Depends(require_any_role)],
)
async def delete_conversation(
    conversation_id: str,
    current_user: Annotated[UserResponse, Depends(get_current_active_user)],
    db: Annotated[AsyncDatabase, Depends(get_db)],
) -> dict:
    """
    Delete a conversation.

    Both patients and admins can use this endpoint.
    Users can only delete their own conversations.
    """
    try:
        from bson import ObjectId
        obj_id = ObjectId(conversation_id)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )

    result = await db[CHAT_HISTORY_COLLECTION].delete_one({
        "_id": obj_id,
        "user_id": current_user.id
    })

    if result.deleted_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )

    logger.info("Conversation deleted: %s by user: %s", conversation_id, current_user.email)

    return {"message": "Conversation deleted successfully", "conversation_id": conversation_id}


def generate_ai_response(user_message: str, user_role: str) -> str:
    """
    Generate AI response based on user message and role.

    This is a placeholder implementation. In production, this would
    integrate with an actual AI service (e.g., OpenAI, Anthropic, local LLM).
    """
    # Simple keyword-based responses for demo
    message_lower = user_message.lower()

    # Medical disclaimer for all responses
    disclaimer = "\n\n---\n*Disclaimer: This AI assistant provides general health information only. For medical emergencies, call emergency services immediately. For personalized medical advice, consult with a healthcare professional.*"

    if any(keyword in message_lower for keyword in ["emergency", "urgent", "help", "chest pain", "difficulty breathing", "unconscious"]):
        return ("This sounds like it could be a medical emergency. Please call emergency services (911 in the US) "
                "or go to the nearest emergency room immediately. Do not wait for an AI response.") + disclaimer

    if any(keyword in message_lower for keyword in ["appointment", "book", "schedule", "doctor", "specialist"]):
        return ("I can help you understand how to book appointments at MediTrust. You can schedule consultations "
                "through our website's appointment system or by calling our front desk. Would you like information "
                "about a specific department or specialist?") + disclaimer

    if any(keyword in message_lower for keyword in ["lab", "result", "test", "blood", "report"]):
        return ("I can provide general information about lab tests and what typical ranges mean. However, "
                "lab results should always be interpreted by your healthcare provider who knows your full "
                "medical history. Would you like general information about a specific test?") + disclaimer

    if any(keyword in message_lower for keyword in ["medication", "prescription", "drug", "medicine", "dosage"]):
        return ("I can provide general information about medications, but I cannot provide specific dosage "
                "recommendations or prescription advice. Always follow your doctor's instructions and consult "
                "with a pharmacist for medication questions.") + disclaimer

    if any(keyword in message_lower for keyword in ["symptom", "pain", "fever", "headache", "nausea", "cough"]):
        return ("I can provide general information about symptoms, but I cannot diagnose medical conditions. "
                "If you're experiencing concerning symptoms, please consult with a healthcare professional for "
                "proper evaluation and treatment.") + disclaimer

    if user_role == "admin" and any(keyword in message_lower for keyword in ["admin", "dashboard", "statistics", "user", "document"]):
        return ("As an administrator, you have access to the admin dashboard where you can view system "
                "statistics, manage documents, and monitor user activity. Is there something specific about "
                "the admin panel you'd like help with?") + disclaimer

    # Default response
    responses = [
        "I understand you're asking about that. As an AI health assistant, I can provide general information, but please consult with a healthcare professional for personalized medical advice.",
        "That's a great question. For specific health concerns, I recommend scheduling an appointment with one of our specialists at MediTrust. Would you like me to help you find the right department?",
        "I can provide general health information on that topic. However, for accurate diagnosis and treatment planning, please consult with a qualified healthcare provider.",
        "Thank you for your question. Remember that this AI assistant provides informational support only and is not a substitute for professional medical consultation."
    ]

    import random
    return random.choice(responses) + disclaimer