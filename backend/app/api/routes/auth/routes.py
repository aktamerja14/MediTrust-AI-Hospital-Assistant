"""Authentication API routes."""
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pymongo.asynchronous.database import AsyncDatabase

from app.database.connection import get_db
from app.schemas.auth import Token, UserCreate, UserLogin, UserResponse
from app.services.auth import AuthService
from app.api.dependencies import get_auth_service, get_current_active_user

logger = logging.getLogger("meditrust.api.auth")

router = APIRouter(prefix="/auth", tags=["Authentication"])


# Public endpoints


@router.post(
    "/signup",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new patient account",
    responses={
        201: {"description": "Patient account created successfully"},
        400: {"description": "Email already registered or invalid input"},
        422: {"description": "Validation error"},
    },
)
async def signup(
    user_data: UserCreate,
    auth_service: Annotated[AuthService, Depends(get_auth_service)],
) -> UserResponse:
    """
    Register a new patient account.

    - **email**: Valid email address (used as username)
    - **password**: Minimum 8 characters

    Always assigns 'patient' role. Admin registration is not available via this endpoint.
    """
    try:
        user = await auth_service.create_user(user_data, role="patient")
        return user
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


@router.post(
    "/login",
    response_model=Token,
    summary="Authenticate user and return access token",
    responses={
        200: {"description": "Login successful, returns access token"},
        401: {"description": "Invalid credentials"},
        422: {"description": "Validation error"},
    },
)
async def login(
    login_data: UserLogin,
    auth_service: Annotated[AuthService, Depends(get_auth_service)],
) -> Token:
    """
    Authenticate user with email and password.

    Returns a JWT access token on success.
    Uses generic error message to prevent user enumeration.
    """
    user = await auth_service.authenticate_user(login_data)

    if not user:
        logger.warning("Failed login attempt for email: %s", login_data.email)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = auth_service.create_access_token(user)
    logger.info("User logged in: %s (role: %s)", user.email, user.role)
    return token


# Protected endpoints


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get current authenticated user profile",
    responses={
        200: {"description": "Current user profile"},
        401: {"description": "Not authenticated"},
        404: {"description": "User not found"},
    },
)
async def get_current_user_profile(
    current_user: Annotated[UserResponse, Depends(get_current_active_user)],
) -> UserResponse:
    """
    Get the profile of the currently authenticated user.

    Requires valid JWT access token in Authorization header.
    """
    return current_user