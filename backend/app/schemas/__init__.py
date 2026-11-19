"""Schemas package."""
from app.schemas.auth import (
    TokenPayload,
    UserBase,
    UserCreate,
    UserLogin,
    Token,
    UserResponse,
    UserInDB,
    Message,
)
from app.schemas.doctor import (
    DoctorBase,
    DoctorResponse,
    DoctorListResponse,
    DoctorRecommendation,
)

__all__ = [
    "TokenPayload",
    "UserBase",
    "UserCreate",
    "UserLogin",
    "Token",
    "UserResponse",
    "UserInDB",
    "Message",
    "DoctorBase",
    "DoctorResponse",
    "DoctorListResponse",
    "DoctorRecommendation",
]