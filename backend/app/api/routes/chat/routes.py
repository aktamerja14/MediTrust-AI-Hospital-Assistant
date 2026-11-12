"""AI Chatbot API routes - accessible by both patients and admins."""
import logging
from typing import Annotated, Optional, List
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from pymongo.asynchronous.database import AsyncDatabase

from app.database.connection import get_db
from app.api.dependencies import get_auth_service, get_current_active_user, require_any_role
from app.services.auth import AuthService
from app.schemas.auth import UserResponse
from app.ai.rag.chain import RAGChain, RAGChainError

logger = logging.getLogger("meditrust.api.chat")

router = APIRouter(prefix="/chat", tags=["AI Chatbot"])

CHAT_HISTORY_COLLECTION = "chat_history"


class ChatMessage(BaseModel):
    """Chat message structure."""
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Message timestamp")


class SourceReference(BaseModel):
    """Source document reference in RAG response."""
    document_id: str
    source: str
    page: str
    score: float
    content_preview: str


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
    sources: List[SourceReference] = Field(default_factory=list, description="Source document references")
    has_context: bool = Field(default=False, description="Whether relevant context was found")
    model: str = Field(default="", description="LLM model used")


class ConversationListItem(BaseModel):
    """Conversation summary for listing."""
    id: str
    title: str
    preview: str
    updated_at: datetime


class ConversationListResponse(BaseModel):
    """Response for listing conversations."""
    conversations: list[ConversationListItem]


# Singleton RAG chain
_rag_chain: RAGChain = None


def get_rag_chain() -> RAGChain:
    """Get or create RAG chain singleton."""
    global _rag_chain
    if _rag_chain is None:
        _rag_chain = RAGChain()
    return _rag_chain


@router.post(
    "/message",
    response_model=ChatResponse,
    summary="Send a message to the AI chatbot",
    responses={
        200: {"description": "Message processed successfully"},
        401: {"description": "Not authenticated"},
        403: {"description": "Inactive user account"},
        422: {"description": "Validation error"},
        500: {"description": "AI service error"},
    },
    dependencies=[Depends(require_any_role)],
)
async def send_message(
    request: ChatRequest,
    current_user: Annotated[UserResponse, Depends(get_current_active_user)],
    db: Annotated[AsyncDatabase, Depends(get_db)],
) -> ChatResponse:
    """
    Send a message to the AI chatbot and receive a RAG-grounded response.

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

    # Prepare conversation history for RAG (last 6 messages excluding current)
    history = [msg.model_dump() for msg in conversation.messages[:-1]][-6:]

    # Generate AI response using RAG chain
    try:
        rag_chain = get_rag_chain()
        rag_response = rag_chain.generate(
            query=request.message,
            conversation_history=history,
        )
        assistant_content = rag_response.answer
        sources = [SourceReference(**s) for s in rag_response.sources]
        has_context = rag_response.has_context
        model = rag_response.model
    except RAGChainError as e:
        logger.error("RAG chain error: %s", e)
        # Fallback response on RAG failure
        assistant_content = (
            "I apologize, but I'm experiencing technical difficulties accessing the "
            "knowledge base. Please try again in a moment, or contact support if the "
            "issue persists.\n\n"
            "---\n"
            "**Disclaimer:** This AI assistant provides general health information only. "
            "For medical emergencies, call emergency services immediately."
        )
        sources = []
        has_context = False
        model = ""
    except Exception as e:
        logger.exception("Unexpected error in chat: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate response"
        )

    assistant_message = ChatMessage(role="assistant", content=assistant_content)
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

    logger.info("Chat message processed for user: %s (role: %s, has_context: %s)",
                current_user.email, current_user.role, has_context)

    return ChatResponse(
        conversation_id=conversation.id,
        user_message=user_message,
        assistant_message=assistant_message,
        sources=sources,
        has_context=has_context,
        model=model,
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