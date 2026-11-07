"""Authentication request/response schemas."""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr, Field, ConfigDict


class TokenPayload(BaseModel):
    """JWT token payload structure."""
    sub: str
    role: str
    exp: int
    iat: int
    type: str = "access"

    model_config = ConfigDict(extra="allow")


class UserBase(BaseModel):
    """Base user schema with common fields."""
    email: EmailStr = Field(..., description="User email address")


class UserCreate(UserBase):
    """User registration request schema."""
    password: str = Field(
        ...,
        min_length=8,
        max_length=128,
        description="User password (min 8 characters)"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "email": "patient@example.com",
                "password": "securePassword123"
            }
        }
    )


class UserLogin(BaseModel):
    """User login request schema."""
    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., description="User password")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "email": "patient@example.com",
                "password": "securePassword123"
            }
        }
    )


class Token(BaseModel):
    """Token response schema."""
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration time in seconds")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer",
                "expires_in": 1800
            }
        }
    )


class UserResponse(UserBase):
    """User response schema (safe fields only)."""
    id: str = Field(..., description="User unique identifier")
    role: str = Field(..., description="User role (patient, admin)")
    is_active: bool = Field(default=True, description="User active status")
    created_at: datetime = Field(..., description="Account creation timestamp")

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": "507f1f77bcf86cd799439011",
                "email": "patient@example.com",
                "role": "patient",
                "is_active": True,
                "created_at": "2024-01-15T10:30:00Z"
            }
        }
    )


class UserInDB(UserResponse):
    """Internal user schema with hashed password (never exposed via API)."""
    hashed_password: str = Field(..., description="Argon2 hashed password")

    model_config = ConfigDict(from_attributes=True)


class Message(BaseModel):
    """Generic message response."""
    message: str

    model_config = ConfigDict(
        json_schema_extra={"example": {"message": "Operation successful"}}
    )