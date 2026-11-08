"""Authentication business logic service."""
import logging
from datetime import timedelta
from typing import Optional

from pymongo.asynchronous.database import AsyncDatabase
from pymongo.errors import DuplicateKeyError
from bson import ObjectId

from app.core.config import settings
from app.core.security.password import hash_password, verify_password
from app.core.security.jwt import create_access_token
from app.schemas.auth import UserCreate, UserInDB, UserLogin, UserResponse, Token

logger = logging.getLogger("meditrust.services.auth")

USERS_COLLECTION = "users"


class AuthService:
    """Service layer for authentication operations."""

    def __init__(self, db: AsyncDatabase) -> None:
        self.db = db
        self.collection = db[USERS_COLLECTION]

    async def create_user(self, user_data: UserCreate, role: str = "patient") -> UserResponse:
        """
        Create a new user with hashed password.

        Args:
            user_data: User registration data.
            role: User role (default: patient). Admin creation is restricted.

        Returns:
            Created user response (without password).

        Raises:
            ValueError: If email already exists or invalid role.
        """
        if role not in ("patient", "admin"):
            raise ValueError("Invalid role. Must be 'patient' or 'admin'")

        # Check for existing user
        existing = await self.collection.find_one({"email": user_data.email.lower()})
        if existing:
            raise ValueError("Email already registered")

        # Hash password
        hashed_password = hash_password(user_data.password)

        # Prepare user document
        user_doc = {
            "email": user_data.email.lower(),
            "hashed_password": hashed_password,
            "role": role,
            "is_active": True,
        }

        try:
            result = await self.collection.insert_one(user_doc)
            user_doc["_id"] = result.inserted_id
        except DuplicateKeyError:
            raise ValueError("Email already registered")

        logger.info("Created new user: %s (role: %s)", user_data.email, role)
        return UserResponse(
            id=str(user_doc["_id"]),
            email=user_doc["email"],
            role=user_doc["role"],
            is_active=user_doc["is_active"],
            created_at=user_doc["_id"].generation_time,
        )

    async def authenticate_user(self, login_data: UserLogin) -> Optional[UserInDB]:
        """
        Authenticate user with email and password.

        Args:
            login_data: Login credentials.

        Returns:
            UserInDB if authentication successful, None otherwise.
        """
        user_doc = await self.collection.find_one(
            {"email": login_data.email.lower(), "is_active": True}
        )

        if not user_doc:
            return None

        if not verify_password(login_data.password, user_doc["hashed_password"]):
            return None

        return UserInDB(
            id=str(user_doc["_id"]),
            email=user_doc["email"],
            role=user_doc["role"],
            is_active=user_doc["is_active"],
            created_at=user_doc["_id"].generation_time,
            hashed_password=user_doc["hashed_password"],
        )

    async def get_user_by_id(self, user_id: str) -> Optional[UserResponse]:
        """
        Get user by ID.

        Args:
            user_id: User ObjectId string.

        Returns:
            UserResponse if found, None otherwise.
        """
        try:
            obj_id = ObjectId(user_id)
        except Exception:
            return None

        user_doc = await self.collection.find_one({"_id": obj_id, "is_active": True})
        if not user_doc:
            return None

        return UserResponse(
            id=str(user_doc["_id"]),
            email=user_doc["email"],
            role=user_doc["role"],
            is_active=user_doc["is_active"],
            created_at=user_doc["_id"].generation_time,
        )

    async def get_user_by_email(self, email: str) -> Optional[UserInDB]:
        """
        Get user by email (internal use).

        Args:
            email: User email.

        Returns:
            UserInDB if found, None otherwise.
        """
        user_doc = await self.collection.find_one({"email": email.lower(), "is_active": True})
        if not user_doc:
            return None

        return UserInDB(
            id=str(user_doc["_id"]),
            email=user_doc["email"],
            role=user_doc["role"],
            is_active=user_doc["is_active"],
            created_at=user_doc["_id"].generation_time,
            hashed_password=user_doc["hashed_password"],
        )

    def create_access_token(self, user: UserInDB) -> Token:
        """
        Create access token for authenticated user.

        Args:
            user: Authenticated user.

        Returns:
            Token response with access token and metadata.
        """
        expire_minutes = settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES
        access_token = create_access_token(
            subject=user.id,
            role=user.role,
            expires_delta=timedelta(minutes=expire_minutes),
        )
        return Token(
            access_token=access_token,
            token_type="bearer",
            expires_in=expire_minutes * 60,
        )

    async def get_dashboard_stats(self) -> dict:
        """
        Get dashboard statistics.

        Returns:
            Dictionary with user statistics.
        """
        total_users = await self.collection.count_documents({})
        total_patients = await self.collection.count_documents({"role": "patient"})
        total_admins = await self.collection.count_documents({"role": "admin"})
        active_users = await self.collection.count_documents({"is_active": True})
        inactive_users = await self.collection.count_documents({"is_active": False})

        return {
            "total_users": total_users,
            "total_patients": total_patients,
            "total_admins": total_admins,
            "active_users": active_users,
            "inactive_users": inactive_users,
        }