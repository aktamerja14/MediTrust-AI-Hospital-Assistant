"""FastAPI dependencies for authentication and authorization."""
import logging
from typing import Annotated, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pymongo.asynchronous.database import AsyncDatabase

from app.database.connection import get_db
from app.core.config import settings
from app.core.security.jwt import decode_token
from app.schemas.auth import TokenPayload, UserResponse
from app.services.auth import AuthService

logger = logging.getLogger("meditrust.api.dependencies")

http_bearer = HTTPBearer(auto_error=False)


async def get_token_from_header(
    credentials: Annotated[Optional[HTTPAuthorizationCredentials], Depends(http_bearer)],
) -> Optional[str]:
    """
    Extract Bearer token from Authorization header.

    Args:
        credentials: HTTP Bearer credentials from Authorization header.

    Returns:
        Token string if present, None otherwise.
    """
    if credentials is None:
        return None
    return credentials.credentials


async def get_auth_service(
    db: Annotated[AsyncDatabase, Depends(get_db)],
) -> AuthService:
    """Get AuthService instance with database connection."""
    return AuthService(db)


async def get_token_payload(
    token: Annotated[Optional[str], Depends(get_token_from_header)],
) -> TokenPayload:
    """
    Extract and validate JWT token payload.

    Args:
        token: JWT token from Authorization header.

    Returns:
        Validated TokenPayload.

    Raises:
        HTTPException: If token is missing, invalid, or expired.
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if payload.type != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return payload


async def get_current_user(
    token_payload: Annotated[TokenPayload, Depends(get_token_payload)],
    auth_service: Annotated[AuthService, Depends(get_auth_service)],
) -> UserResponse:
    """
    Get current authenticated user from token.

    Args:
        token_payload: Validated JWT payload.
        auth_service: AuthService instance.

    Returns:
        Current user profile.

    Raises:
        HTTPException: If user not found or inactive.
    """
    user = await auth_service.get_user_by_id(token_payload.sub)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


async def get_current_active_user(
    current_user: Annotated[UserResponse, Depends(get_current_user)],
) -> UserResponse:
    """
    Get current active user (additional active status check).

    Args:
        current_user: Current user from token.

    Returns:
        Current active user.

    Raises:
        HTTPException: If user is inactive.
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user account",
        )
    return current_user


def require_role(*allowed_roles: str):
    """
    Dependency factory for role-based access control.

    Args:
        *allowed_roles: Allowed roles (e.g., "admin", "patient").

    Returns:
        Dependency that checks user role.

    Usage:
        @router.get("/admin-only", dependencies=[Depends(require_role("admin"))])
    """
    async def role_checker(
        current_user: Annotated[UserResponse, Depends(get_current_active_user)],
    ) -> UserResponse:
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required role(s): {', '.join(allowed_roles)}",
            )
        return current_user

    return role_checker


# Common role dependencies
require_admin = require_role("admin")
require_patient = require_role("patient")
require_any_role = require_role("patient", "admin")


def verify_patient_access(
    patient_id: str,
    current_user: Annotated[UserResponse, Depends(get_current_active_user)],
) -> UserResponse:
    """
    Verify that a patient can only access their own records.

    Args:
        patient_id: Patient ID from path/query parameter.
        current_user: Current authenticated user.

    Returns:
        Current user if access is allowed.

    Raises:
        HTTPException: If patient tries to access another patient's data.
    """
    if current_user.role == "patient" and current_user.id != patient_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied. Patients can only access their own records.",
        )
    return current_user