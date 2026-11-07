"""JWT token creation and validation utilities."""
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from jose import JWTError, jwt
from pydantic import ValidationError

from app.core.config import settings
from app.schemas.auth import TokenPayload

logger = logging.getLogger("meditrust.security.jwt")


def create_access_token(
    subject: str,
    role: str,
    expires_delta: Optional[timedelta] = None,
    extra_claims: Optional[dict[str, Any]] = None,
) -> str:
    """
    Create a JWT access token.

    Args:
        subject: User identifier (typically user ID).
        role: User role (patient, admin).
        expires_delta: Optional custom expiration timedelta.
        extra_claims: Optional additional claims to include in token.

    Returns:
        Encoded JWT access token string.
    """
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES
        )

    claims: dict[str, Any] = {
        "sub": str(subject),
        "role": role,
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "type": "access",
    }

    if extra_claims:
        claims.update(extra_claims)

    encoded_jwt = jwt.encode(
        claims,
        settings.JWT_SECRET_KEY.get_secret_value() if hasattr(settings.JWT_SECRET_KEY, 'get_secret_value') else settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )
    return encoded_jwt


def decode_token(token: str) -> Optional[TokenPayload]:
    """
    Decode and validate a JWT access token.

    Args:
        token: JWT token string.

    Returns:
        TokenPayload if valid, None otherwise.
    """
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY.get_secret_value() if hasattr(settings.JWT_SECRET_KEY, 'get_secret_value') else settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
        return TokenPayload(**payload)
    except JWTError as exc:
        logger.warning("JWT decode error: %s", exc)
        return None
    except ValidationError as exc:
        logger.warning("JWT validation error: %s", exc)
        return None


def get_token_expiration(token: str) -> Optional[datetime]:
    """
    Get expiration datetime from a JWT token without full validation.

    Args:
        token: JWT token string.

    Returns:
        Expiration datetime if token has exp claim, None otherwise.
    """
    try:
        payload = jwt.get_unverified_claims(token)
        exp = payload.get("exp")
        if exp:
            return datetime.fromtimestamp(exp, tz=timezone.utc)
    except JWTError:
        pass
    return None