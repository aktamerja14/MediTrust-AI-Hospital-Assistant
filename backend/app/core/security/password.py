"""Argon2 password hashing utilities."""
import logging
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, VerificationError

from app.core.config import settings

logger = logging.getLogger("meditrust.security.password")

_hasher = PasswordHasher(
    time_cost=settings.ARGON2_TIME_COST,
    memory_cost=settings.ARGON2_MEMORY_COST,
    parallelism=settings.ARGON2_PARALLELISM,
    hash_len=settings.ARGON2_HASH_LEN,
    salt_len=settings.ARGON2_SALT_LEN,
)


def hash_password(password: str) -> str:
    """
    Hash a plain-text password using Argon2id.

    Args:
        password: Plain-text password to hash.

    Returns:
        Argon2 hash string containing algorithm parameters, salt, and hash.
    """
    if not password:
        raise ValueError("Password cannot be empty")
    return _hasher.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plain-text password against an Argon2 hash.

    Args:
        plain_password: Plain-text password to verify.
        hashed_password: Argon2 hash string from database.

    Returns:
        True if password matches, False otherwise.
    """
    if not plain_password or not hashed_password:
        return False
    try:
        _hasher.verify(hashed_password, plain_password)
        return True
    except VerifyMismatchError:
        return False
    except VerificationError as exc:
        logger.error("Password verification error: %s", exc)
        return False


def needs_rehash(hashed_password: str) -> bool:
    """
    Check if a hashed password needs rehashing with current parameters.

    Args:
        hashed_password: Existing Argon2 hash string.

    Returns:
        True if rehashing is recommended, False otherwise.
    """
    if not hashed_password:
        return True
    try:
        return _hasher.check_needs_rehash(hashed_password)
    except Exception:
        return True