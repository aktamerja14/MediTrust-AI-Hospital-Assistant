import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from pymongo import AsyncMongoClient
from pymongo.asynchronous.database import AsyncDatabase
from pymongo.errors import PyMongoError

from app.core.config import settings

logger = logging.getLogger("meditrust.database")


class DatabaseManager:
    """Manages asynchronous MongoDB connection lifecycle via pymongo AsyncMongoClient."""

    def __init__(self) -> None:
        self._client: AsyncMongoClient | None = None
        self._db: AsyncDatabase | None = None

    @property
    def client(self) -> AsyncMongoClient | None:
        return self._client

    @property
    def db(self) -> AsyncDatabase | None:
        return self._db

    async def connect(self) -> None:
        """Initialize AsyncMongoClient and connect to MongoDB."""
        if self._client is not None:
            logger.warning("Database connection already initialized")
            return

        try:
            masked_url = settings.MONGODB_URL.split("@")[-1] if "@" in settings.MONGODB_URL else settings.MONGODB_URL
            logger.info("Connecting to MongoDB at %s...", masked_url)
            self._client = AsyncMongoClient(
                settings.MONGODB_URL,
                serverSelectionTimeoutMS=5000,
            )
            self._db = self._client[settings.DATABASE_NAME]
            await self._client.admin.command("ping")
            logger.info("Connected to MongoDB database: %s", settings.DATABASE_NAME)
        except PyMongoError as exc:
            logger.error("Failed to connect to MongoDB: %s", exc)
            self._client = None
            self._db = None
            raise
        except Exception as exc:
            logger.error("Unexpected error during MongoDB connection: %s", exc)
            self._client = None
            self._db = None
            raise

    async def disconnect(self) -> None:
        """Close AsyncMongoClient connection."""
        if self._client is None:
            logger.warning("Database connection not initialized, nothing to close")
            return

        try:
            logger.info("Closing MongoDB connection...")
            await self._client.close()
            logger.info("MongoDB connection closed.")
        except PyMongoError as exc:
            logger.error("Error closing MongoDB connection: %s", exc)
        finally:
            self._client = None
            self._db = None

    async def ping(self) -> bool:
        """Ping the database to verify connectivity."""
        if self._client is None:
            return False
        try:
            await self._client.admin.command("ping")
            return True
        except PyMongoError:
            return False


db_manager = DatabaseManager()


async def get_db() -> AsyncDatabase | None:
    """FastAPI dependency to provide access to database instance."""
    return db_manager.db


@asynccontextmanager
async def lifespan(app) -> AsyncGenerator[None, None]:
    """FastAPI lifespan context manager for database connection lifecycle."""
    try:
        await db_manager.connect()
    except Exception as exc:
        logger.critical("Failed to initialize database connection on startup: %s", exc)
        raise
    yield
    await db_manager.disconnect()