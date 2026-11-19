"""AI Chatbot API routes - accessible by both patients and admins."""
import logging
import uuid
from typing import Annotated, Optional, List
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from pydantic import BaseModel, Field
from pymongo.asynchronous.database import AsyncDatabase
from bson import ObjectId

from app.database.connection import get_db
from app.api.dependencies import get_auth_service, get_current_active_user, require_any_role
from app.services.auth import AuthService
from app.schemas.auth import UserResponse
from app.schemas.doctor import DoctorRecommendation
from app.ai.rag.chain import RAGChain, RAGChainError
from app.ai.rag.prompts import MEDICAL_DISCLAIMER

logger = logging.getLogger("meditrust.api.chat")

router = APIRouter(prefix="/chat", tags=["AI Chatbot"])

CHAT_HISTORY_COLLECTION = "chat_history"
MESSAGE_FEEDBACK_COLLECTION = "message_feedback"
CONVERSATION_RATING_COLLECTION = "conversation_ratings"


def _utc_now() -> datetime:
    """Return current UTC time as timezone-aware datetime."""
    return datetime.now(timezone.utc)


def _ensure_utc(dt: datetime) -> datetime:
    """Ensure datetime is timezone-aware UTC. MongoDB stores naive UTC datetimes."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _generate_feedback_uuid() -> str:
    """Generate a stable UUID for LangSmith feedback tracking."""
    return str(uuid.uuid4())


async def _sync_message_feedback_to_langsmith(
    run_id: Optional[str],
    trace_id: Optional[str],
    score: float,
    feedback_id: str,
) -> None:
    """
    Background task to sync message feedback to LangSmith.

    Fire-and-forget: MongoDB is source of truth. LangSmith failures are logged
    but never affect the API response or MongoDB feedback.

    Privacy: Only sends run_id/trace_id, fixed key, numeric score, feedback UUID.
    Never sends user IDs, message IDs, comments, queries, answers, or medical data.
    """
    # Skip if no run_id (tracing was disabled when message was created)
    if not run_id:
        logger.debug("Skipping LangSmith sync: no run_id provided")
        return

    try:
        from langsmith import Client
        from langsmith.schemas import FeedbackSourceType

        client = Client()
        client.create_feedback(
            run_id=run_id,
            trace_id=trace_id,
            key="user_feedback",
            score=score,
            feedback_id=feedback_id,
            feedback_source_type=FeedbackSourceType.API,
        )
        logger.info("LangSmith message feedback synced: run_id=%s, score=%s", run_id, score)
    except Exception as e:
        logger.warning("LangSmith message feedback sync failed: %s", type(e).__name__)


async def _sync_conversation_rating_to_langsmith(
    trace_id: str,
    score: float,
    feedback_id: str,
) -> None:
    """
    Background task to sync conversation rating to LangSmith.

    Uses root trace_id (not child runs). MongoDB is source of truth.
    LangSmith failures are logged but never affect the API response or MongoDB rating.

    Privacy: Only sends trace_id, fixed key, normalized score, feedback UUID.
    Never sends user IDs, conversation IDs, comments, or medical data.
    """
    try:
        from langsmith import Client
        from langsmith.schemas import FeedbackSourceType

        client = Client()
        client.create_feedback(
            trace_id=trace_id,
            key="conversation_rating",
            score=score,
            feedback_id=feedback_id,
            feedback_source_type=FeedbackSourceType.API,
        )
        logger.info("LangSmith conversation rating synced: trace_id=%s, score=%s", trace_id, score)
    except Exception as e:
        logger.warning("LangSmith conversation rating sync failed: %s", type(e).__name__)


class ChatMessage(BaseModel):
    """Chat message structure."""
    id: str = Field(..., description="Message unique identifier")
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=_utc_now, description="Message timestamp")
    doctor_recommendations: List[DoctorRecommendation] = Field(default_factory=list, description="Recommended doctors for medical questions")
    langsmith_run_id: Optional[str] = Field(default=None, description="LangSmith run ID for tracing")
    langsmith_trace_id: Optional[str] = Field(default=None, description="LangSmith trace ID for tracing")


class SourceReference(BaseModel):
    """Source document reference in RAG response."""
    document_id: str
    source: str
    page: str
    score: float
    content_preview: str


class DoctorRecommendation(BaseModel):
    """Doctor recommendation in chat response."""
    doctor_id: str
    name: str
    qualifications: str
    specialization: str
    experience_years: int
    photo_path: str
    phone: str
    available_from: str
    available_to: str
    match_reason: str


class ChatConversation(BaseModel):
    """Chat conversation structure."""
    id: str = Field(..., description="Conversation unique identifier")
    user_id: str = Field(..., description="User ID who owns this conversation")
    title: str = Field(..., description="Conversation title")
    messages: list[ChatMessage] = Field(default_factory=list, description="List of messages")
    created_at: datetime = Field(default_factory=_utc_now, description="Conversation creation timestamp")
    updated_at: datetime = Field(default_factory=_utc_now, description="Last update timestamp (title, settings)")
    last_message_at: datetime = Field(default_factory=_utc_now, description="Last message timestamp")


class ChatRequest(BaseModel):
    """Request to send a message in a conversation."""
    message: str = Field(..., min_length=1, max_length=4000, description="User message")
    conversation_id: Optional[str] = Field(None, description="Existing conversation ID (optional)")


class ConversationRenameRequest(BaseModel):
    """Request to rename a conversation."""
    title: str = Field(..., min_length=1, max_length=200, description="New conversation title")


class ChatResponse(BaseModel):
    """Response from chatbot."""
    conversation_id: str = Field(..., description="Conversation ID")
    user_message: ChatMessage = Field(..., description="Echo of user message")
    assistant_message: ChatMessage = Field(..., description="AI assistant response")
    sources: List[SourceReference] = Field(default_factory=list, description="Source document references")
    has_context: bool = Field(default=False, description="Whether relevant context was found")
    model: str = Field(default="", description="LLM model used")
    doctor_recommendations: List[DoctorRecommendation] = Field(default_factory=list, description="Recommended doctors for medical questions")
    disclaimer: str = Field(default="", description="Medical disclaimer if applicable")


class ConversationListItem(BaseModel):
    """Conversation summary for listing."""
    id: str
    title: str
    preview: str
    last_message_at: datetime
    updated_at: datetime


class ConversationListResponse(BaseModel):
    """Response for listing conversations."""
    conversations: list[ConversationListItem]


class MessageFeedbackRequest(BaseModel):
    """Request to submit feedback for an assistant message."""
    conversation_id: str = Field(..., description="Conversation ID")
    message_id: str = Field(..., description="Assistant message ID")
    feedback_type: str = Field(..., description="Feedback type: 'positive' or 'negative'")


class MessageFeedbackResponse(BaseModel):
    """Response for message feedback."""
    id: str
    user_id: str
    conversation_id: str
    message_id: str
    feedback_type: str
    created_at: datetime


class ConversationRatingRequest(BaseModel):
    """Request to submit a star rating for a conversation."""
    rating: int = Field(..., ge=1, le=5, description="Star rating (1-5)")


class ConversationRatingResponse(BaseModel):
    """Response for conversation rating."""
    id: str
    user_id: str
    conversation_id: str
    rating: int
    created_at: datetime


class ConversationRatingStatusResponse(BaseModel):
    """Response for checking if conversation has been rated."""
    rated: bool
    rating: Optional[int] = None


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
            conversation_doc = await db[CHAT_HISTORY_COLLECTION].find_one({
                "_id": ObjectId(request.conversation_id),
                "user_id": current_user.id
            })
            if conversation_doc:
                raw_messages = conversation_doc.get("messages", [])
                messages = []
                for msg in raw_messages:
                    msg_copy = dict(msg)
                    if "timestamp" in msg_copy:
                        msg_copy["timestamp"] = _ensure_utc(msg_copy["timestamp"])
                    if "id" not in msg_copy:
                        msg_copy["id"] = str(ObjectId())
                    messages.append(ChatMessage(**msg_copy))
                conversation = ChatConversation(
                    id=str(conversation_doc["_id"]),
                    user_id=conversation_doc["user_id"],
                    title=conversation_doc["title"],
                    messages=messages,
                    created_at=_ensure_utc(conversation_doc["created_at"]),
                    updated_at=_ensure_utc(conversation_doc["updated_at"]),
                    last_message_at=_ensure_utc(conversation_doc.get("last_message_at", conversation_doc["updated_at"])),
                )
        except Exception:
            pass

    if not conversation:
        # Create new conversation
        now = _utc_now()
        conversation = ChatConversation(
            id="",  # Will be set after insert
            user_id=current_user.id,
            title=request.message[:50] + ("..." if len(request.message) > 50 else ""),
            messages=[],
            created_at=now,
            updated_at=now,
            last_message_at=now,
        )

    # Add user message
    user_message = ChatMessage(id=str(ObjectId()), role="user", content=request.message)
    conversation.messages.append(user_message)

    # Prepare conversation history for RAG (last 6 messages excluding current)
    history = [msg.model_dump() for msg in conversation.messages[:-1]][-6:]

    # Generate AI response using RAG chain
    try:
        rag_chain = get_rag_chain()
        rag_response = await rag_chain.generate(
            query=request.message,
            conversation_history=history,
            db=db,
        )
        assistant_content = rag_response.answer
        sources = [SourceReference(**s) for s in rag_response.sources]
        has_context = rag_response.has_context
        model = rag_response.model
        doctor_recommendations = rag_response.doctor_recommendations or []
        disclaimer = rag_response.disclaimer
        langsmith_run_id = rag_response.langsmith_run_id
        langsmith_trace_id = rag_response.langsmith_trace_id
    except RAGChainError as e:
        logger.error("RAG chain error: %s", e)
        # Fallback response on RAG failure
        assistant_content = (
            "I apologize, but I'm experiencing technical difficulties accessing the "
            "knowledge base. Please try again in a moment, or contact support if the "
            "issue persists."
        )
        sources = []
        has_context = False
        model = ""
        doctor_recommendations = []
        disclaimer = MEDICAL_DISCLAIMER
        langsmith_run_id = None
        langsmith_trace_id = None
    except Exception as e:
        logger.exception("Unexpected error in chat: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate response"
        )

    assistant_message = ChatMessage(
        id=str(ObjectId()),
        role="assistant",
        content=assistant_content,
        doctor_recommendations=doctor_recommendations,
        langsmith_run_id=langsmith_run_id,
        langsmith_trace_id=langsmith_trace_id,
    )
    conversation.messages.append(assistant_message)

    # Update timestamps
    now = _utc_now()
    conversation.updated_at = now
    conversation.last_message_at = now

    # Save to database
    conversation_doc = {
        "user_id": conversation.user_id,
        "title": conversation.title,
        "messages": [msg.model_dump() for msg in conversation.messages],
        "created_at": conversation.created_at,
        "updated_at": conversation.updated_at,
        "last_message_at": conversation.last_message_at,
    }

    if conversation.id:
        # Update existing
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
        doctor_recommendations=doctor_recommendations,
        disclaimer=disclaimer,
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
    Returns conversations sorted by most recent message first.
    """
    cursor = db[CHAT_HISTORY_COLLECTION].find(
        {"user_id": current_user.id},
        {"title": 1, "messages": 1, "last_message_at": 1, "created_at": 1, "updated_at": 1}
    ).sort("last_message_at", -1)

    conversations = []
    async for doc in cursor:
        messages = doc.get("messages", [])
        preview = ""
        if messages:
            last_msg = messages[-1]
            preview = last_msg.get("content", "")[:100]
            if len(last_msg.get("content", "")) > 100:
                preview += "..."

        last_msg_at = doc.get("last_message_at", doc.get("updated_at", doc.get("created_at")))
        updated_at = doc.get("updated_at", doc.get("created_at"))

        conversations.append(ConversationListItem(
            id=str(doc["_id"]),
            title=doc.get("title", "Untitled"),
            preview=preview,
            last_message_at=_ensure_utc(last_msg_at) if last_msg_at else _utc_now(),
            updated_at=_ensure_utc(updated_at) if updated_at else _utc_now(),
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

    raw_messages = conversation_doc.get("messages", [])
    messages = []
    messages_updated = False
    for msg in raw_messages:
        msg_copy = dict(msg)
        if "timestamp" in msg_copy:
            msg_copy["timestamp"] = _ensure_utc(msg_copy["timestamp"])
        if "id" not in msg_copy:
            msg_copy["id"] = str(ObjectId())
            messages_updated = True
        messages.append(ChatMessage(**msg_copy))

    if messages_updated:
        await db[CHAT_HISTORY_COLLECTION].update_one(
            {"_id": obj_id},
            {"$set": {"messages": [msg.model_dump() for msg in messages]}}
        )
        logger.info("[CHAT] Persisted generated message IDs for conversation: %s", conversation_id)

    return ChatConversation(
        id=str(conversation_doc["_id"]),
        user_id=conversation_doc["user_id"],
        title=conversation_doc["title"],
        messages=messages,
        created_at=_ensure_utc(conversation_doc["created_at"]),
        updated_at=_ensure_utc(conversation_doc["updated_at"]),
        last_message_at=_ensure_utc(conversation_doc.get("last_message_at", conversation_doc.get("updated_at", conversation_doc["created_at"]))),
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


@router.patch(
    "/conversations/{conversation_id}",
    response_model=ChatConversation,
    summary="Rename a conversation",
    responses={
        200: {"description": "Conversation renamed successfully"},
        401: {"description": "Not authenticated"},
        403: {"description": "Inactive user account or access denied"},
        404: {"description": "Conversation not found"},
        422: {"description": "Validation error"},
    },
    dependencies=[Depends(require_any_role)],
)
async def rename_conversation(
    conversation_id: str,
    request: ConversationRenameRequest,
    current_user: Annotated[UserResponse, Depends(get_current_active_user)],
    db: Annotated[AsyncDatabase, Depends(get_db)],
) -> ChatConversation:
    """
    Rename a conversation.

    Both patients and admins can use this endpoint.
    Users can only rename their own conversations.
    """
    try:
        obj_id = ObjectId(conversation_id)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )

    # Update title and updated_at
    now = _utc_now()
    result = await db[CHAT_HISTORY_COLLECTION].update_one(
        {"_id": obj_id, "user_id": current_user.id},
        {"$set": {"title": request.title, "updated_at": now}}
    )

    if result.matched_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )

    # Fetch updated conversation
    conversation_doc = await db[CHAT_HISTORY_COLLECTION].find_one({
        "_id": obj_id,
        "user_id": current_user.id
    })

    logger.info("Conversation renamed: %s by user: %s", conversation_id, current_user.email)

    raw_messages = conversation_doc.get("messages", [])
    messages = []
    messages_updated = False
    for msg in raw_messages:
        msg_copy = dict(msg)
        if "timestamp" in msg_copy:
            msg_copy["timestamp"] = _ensure_utc(msg_copy["timestamp"])
        if "id" not in msg_copy:
            msg_copy["id"] = str(ObjectId())
            messages_updated = True
        messages.append(ChatMessage(**msg_copy))

    if messages_updated:
        await db[CHAT_HISTORY_COLLECTION].update_one(
            {"_id": obj_id},
            {"$set": {"messages": [msg.model_dump() for msg in messages]}}
        )
        logger.info("[CHAT] Persisted generated message IDs for conversation: %s", conversation_id)

    return ChatConversation(
        id=str(conversation_doc["_id"]),
        user_id=conversation_doc["user_id"],
        title=conversation_doc["title"],
        messages=messages,
        created_at=_ensure_utc(conversation_doc["created_at"]),
        updated_at=_ensure_utc(conversation_doc["updated_at"]),
        last_message_at=_ensure_utc(conversation_doc.get("last_message_at", conversation_doc.get("updated_at", conversation_doc["created_at"]))),
    )


@router.post(
    "/feedback",
    response_model=MessageFeedbackResponse,
    summary="Submit feedback for an assistant message",
    responses={
        200: {"description": "Feedback submitted successfully"},
        201: {"description": "Feedback created successfully"},
        400: {"description": "Invalid feedback type"},
        401: {"description": "Not authenticated"},
        403: {"description": "Inactive user account or access denied"},
        404: {"description": "Conversation or message not found"},
        409: {"description": "Feedback already exists for this message"},
    },
    dependencies=[Depends(require_any_role)],
)
async def submit_message_feedback(
    request: MessageFeedbackRequest,
    current_user: Annotated[UserResponse, Depends(get_current_active_user)],
    db: Annotated[AsyncDatabase, Depends(get_db)],
    background_tasks: BackgroundTasks,
) -> MessageFeedbackResponse:
    """
    Submit thumbs up/down feedback for an assistant message.

    Both patients and admins can use this endpoint.
    Users can only submit feedback for their own conversations.
    Only one feedback per message per user is allowed.
    """
    logger.info("[FEEDBACK] Submit request: conversation_id=%s, message_id=%s, feedback_type=%s, user=%s",
                request.conversation_id, request.message_id, request.feedback_type, current_user.email)

    if request.feedback_type not in ("positive", "negative"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="feedback_type must be 'positive' or 'negative'"
        )

    try:
        obj_id = ObjectId(request.conversation_id)
    except Exception:
        logger.warning("[FEEDBACK] Invalid conversation_id format: %s", request.conversation_id)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )

    conversation_doc = await db[CHAT_HISTORY_COLLECTION].find_one({
        "_id": obj_id,
        "user_id": current_user.id
    })

    if not conversation_doc:
        logger.warning("[FEEDBACK] Conversation not found: %s for user %s", request.conversation_id, current_user.email)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )

    message_ids_in_conversation = [msg.get("id") for msg in conversation_doc.get("messages", [])]
    logger.debug("[FEEDBACK] Conversation %s message IDs: %s", request.conversation_id, message_ids_in_conversation)

    message_exists = False
    matched_message = None
    for msg in conversation_doc.get("messages", []):
        if msg.get("id") == request.message_id and msg.get("role") == "assistant":
            message_exists = True
            matched_message = msg
            break

    if not message_exists:
        logger.warning("[FEEDBACK] Message not found or not assistant: message_id=%s, conversation_id=%s, available_ids=%s",
                       request.message_id, request.conversation_id, message_ids_in_conversation)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Message not found or not an assistant message"
        )

    logger.info("[FEEDBACK] Message found: id=%s, role=%s, content_preview=%s",
                matched_message.get("id"), matched_message.get("role"), matched_message.get("content", "")[:50])

    now = _utc_now()
    existing_feedback = await db[MESSAGE_FEEDBACK_COLLECTION].find_one({
        "user_id": current_user.id,
        "conversation_id": request.conversation_id,
        "message_id": request.message_id
    })

    if existing_feedback:
        # Update existing feedback - reuse the same langsmith_feedback_id
        langsmith_feedback_id = existing_feedback.get("langsmith_feedback_id") or _generate_feedback_uuid()
        await db[MESSAGE_FEEDBACK_COLLECTION].update_one(
            {"_id": existing_feedback["_id"]},
            {"$set": {
                "feedback_type": request.feedback_type,
                "created_at": now,
                "langsmith_feedback_id": langsmith_feedback_id
            }}
        )
        feedback_doc = await db[MESSAGE_FEEDBACK_COLLECTION].find_one({"_id": existing_feedback["_id"]})
    else:
        # New feedback - generate stable UUID for LangSmith tracking
        langsmith_feedback_id = _generate_feedback_uuid()
        feedback_doc = {
            "user_id": current_user.id,
            "conversation_id": request.conversation_id,
            "message_id": request.message_id,
            "feedback_type": request.feedback_type,
            "created_at": now,
            "langsmith_feedback_id": langsmith_feedback_id
        }
        result = await db[MESSAGE_FEEDBACK_COLLECTION].insert_one(feedback_doc)
        feedback_doc["_id"] = result.inserted_id

    logger.info("Message feedback submitted: %s for message %s by user %s",
                request.feedback_type, request.message_id, current_user.email)

    # Fire background sync to LangSmith (non-blocking, MongoDB is source of truth)
    run_id = matched_message.get("langsmith_run_id")
    trace_id = matched_message.get("langsmith_trace_id")
    if run_id:
        score = 1.0 if request.feedback_type == "positive" else 0.0
        background_tasks.add_task(
            _sync_message_feedback_to_langsmith,
            run_id=run_id,
            trace_id=trace_id,
            score=score,
            feedback_id=langsmith_feedback_id,
        )
    else:
        logger.debug("Skipping LangSmith sync: assistant message has no langsmith_run_id")

    return MessageFeedbackResponse(
        id=str(feedback_doc["_id"]),
        user_id=feedback_doc["user_id"],
        conversation_id=feedback_doc["conversation_id"],
        message_id=feedback_doc["message_id"],
        feedback_type=feedback_doc["feedback_type"],
        created_at=_ensure_utc(feedback_doc["created_at"]),
    )


@router.get(
    "/feedback/{message_id}",
    response_model=Optional[MessageFeedbackResponse],
    summary="Get feedback for a specific message",
    responses={
        200: {"description": "Feedback retrieved (or null if not submitted)"},
        401: {"description": "Not authenticated"},
        403: {"description": "Inactive user account or access denied"},
        404: {"description": "Conversation not found"},
    },
    dependencies=[Depends(require_any_role)],
)
async def get_message_feedback(
    message_id: str,
    conversation_id: str,
    current_user: Annotated[UserResponse, Depends(get_current_active_user)],
    db: Annotated[AsyncDatabase, Depends(get_db)],
) -> Optional[MessageFeedbackResponse]:
    """
    Get the current user's feedback for a specific assistant message.

    Returns null if no feedback has been submitted yet.
    """
    try:
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

    feedback_doc = await db[MESSAGE_FEEDBACK_COLLECTION].find_one({
        "user_id": current_user.id,
        "conversation_id": conversation_id,
        "message_id": message_id
    })

    if not feedback_doc:
        return None

    return MessageFeedbackResponse(
        id=str(feedback_doc["_id"]),
        user_id=feedback_doc["user_id"],
        conversation_id=feedback_doc["conversation_id"],
        message_id=feedback_doc["message_id"],
        feedback_type=feedback_doc["feedback_type"],
        created_at=_ensure_utc(feedback_doc["created_at"]),
    )


@router.post(
    "/conversations/{conversation_id}/rate",
    response_model=ConversationRatingResponse,
    summary="Submit a star rating for a conversation",
    responses={
        200: {"description": "Rating submitted successfully"},
        201: {"description": "Rating created successfully"},
        400: {"description": "Invalid rating value"},
        401: {"description": "Not authenticated"},
        403: {"description": "Inactive user account or access denied"},
        404: {"description": "Conversation not found"},
        409: {"description": "Conversation already rated"},
    },
    dependencies=[Depends(require_any_role)],
)
async def submit_conversation_rating(
    conversation_id: str,
    request: ConversationRatingRequest,
    current_user: Annotated[UserResponse, Depends(get_current_active_user)],
    db: Annotated[AsyncDatabase, Depends(get_db)],
    background_tasks: BackgroundTasks,
) -> ConversationRatingResponse:
    """
    Submit a 1-5 star rating for a conversation.

    Both patients and admins can use this endpoint.
    Users can only rate their own conversations.
    Only one rating per conversation per user is allowed.
    """
    if request.rating < 1 or request.rating > 5:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Rating must be between 1 and 5"
        )

    try:
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

    existing_rating = await db[CONVERSATION_RATING_COLLECTION].find_one({
        "user_id": current_user.id,
        "conversation_id": conversation_id
    })

    if existing_rating:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Conversation already rated"
        )

    # Find root trace_id from conversation (first assistant message's trace_id)
    trace_id = None
    for msg in conversation_doc.get("messages", []):
        if msg.get("role") == "assistant" and msg.get("langsmith_trace_id"):
            trace_id = msg.get("langsmith_trace_id")
            break

    now = _utc_now()
    langsmith_feedback_id = _generate_feedback_uuid()
    rating_doc = {
        "user_id": current_user.id,
        "conversation_id": conversation_id,
        "rating": request.rating,
        "created_at": now,
        "langsmith_feedback_id": langsmith_feedback_id
    }
    result = await db[CONVERSATION_RATING_COLLECTION].insert_one(rating_doc)
    rating_doc["_id"] = result.inserted_id

    logger.info("Conversation rating submitted: %d stars for conversation %s by user %s",
                request.rating, conversation_id, current_user.email)

    # Fire background sync to LangSmith (non-blocking, MongoDB is source of truth)
    # Uses root trace_id (not child runs). Normalized score = rating / 5.0
    if trace_id:
        score = request.rating / 5.0
        background_tasks.add_task(
            _sync_conversation_rating_to_langsmith,
            trace_id=trace_id,
            score=score,
            feedback_id=langsmith_feedback_id,
        )
    else:
        logger.debug("Skipping LangSmith sync: conversation has no langsmith_trace_id")

    return ConversationRatingResponse(
        id=str(rating_doc["_id"]),
        user_id=rating_doc["user_id"],
        conversation_id=rating_doc["conversation_id"],
        rating=rating_doc["rating"],
        created_at=_ensure_utc(rating_doc["created_at"]),
    )


@router.get(
    "/conversations/{conversation_id}/rate",
    response_model=ConversationRatingStatusResponse,
    summary="Check if a conversation has been rated",
    responses={
        200: {"description": "Rating status retrieved"},
        401: {"description": "Not authenticated"},
        403: {"description": "Inactive user account or access denied"},
        404: {"description": "Conversation not found"},
    },
    dependencies=[Depends(require_any_role)],
)
async def get_conversation_rating_status(
    conversation_id: str,
    current_user: Annotated[UserResponse, Depends(get_current_active_user)],
    db: Annotated[AsyncDatabase, Depends(get_db)],
) -> ConversationRatingStatusResponse:
    """
    Check if the current user has rated a conversation.

    Returns rated=true with rating value if rated, rated=false otherwise.
    """
    try:
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

    rating_doc = await db[CONVERSATION_RATING_COLLECTION].find_one({
        "user_id": current_user.id,
        "conversation_id": conversation_id
    })

    if rating_doc:
        return ConversationRatingStatusResponse(
            rated=True,
            rating=rating_doc["rating"]
        )

    return ConversationRatingStatusResponse(rated=False)


@router.post(
    "/conversations/{conversation_id}/dismiss-rating",
    response_model=dict,
    summary="Dismiss the rating prompt for a conversation",
    responses={
        200: {"description": "Rating prompt dismissed"},
        401: {"description": "Not authenticated"},
        403: {"description": "Inactive user account or access denied"},
        404: {"description": "Conversation not found"},
    },
    dependencies=[Depends(require_any_role)],
)
async def dismiss_conversation_rating(
    conversation_id: str,
    current_user: Annotated[UserResponse, Depends(get_current_active_user)],
    db: Annotated[AsyncDatabase, Depends(get_db)],
) -> dict:
    """
    Dismiss the rating prompt for a conversation (user clicked Skip).

    This records that the user has explicitly declined to rate,
    so the rating modal will not be shown again for this conversation.
    """
    try:
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

    now = _utc_now()
    await db[CONVERSATION_RATING_COLLECTION].update_one(
        {"user_id": current_user.id, "conversation_id": conversation_id},
        {"$set": {"rating": 0, "dismissed": True, "created_at": now}},
        upsert=True
    )

    logger.info("Conversation rating dismissed for conversation %s by user %s",
                conversation_id, current_user.email)

    return {"message": "Rating prompt dismissed"}
