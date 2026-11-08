"""Test configuration and fixtures for MediTrust backend tests."""
import os
import pytest
import pytest_asyncio
from typing import AsyncGenerator

os.environ["ENVIRONMENT"] = "testing"
os.environ["DATABASE_NAME"] = "meditrust_test"
os.environ["MONGODB_URL"] = "mongodb+srv://ektamerja1411_db_user:Yo4uyoaOubmIStOa@expense-tracker.ks3eflb.mongodb.net/?appName=expense-tracker"

from app.core.config import settings
from app.database.connection import DatabaseManager, get_db
from app.main import create_application
from app.api.dependencies import get_db as get_db_dep
from httpx import AsyncClient, ASGITransport


TEST_DATABASE_NAME = "meditrust_test"
TEST_MONGODB_URL = "mongodb+srv://ektamerja1411_db_user:Yo4uyoaOubmIStOa@expense-tracker.ks3eflb.mongodb.net/?appName=expense-tracker"
DEV_DATABASE_NAME = "meditrust"


def verify_test_database() -> None:
    """Safety check: ensure tests only run against the test database."""
    if settings.DATABASE_NAME != TEST_DATABASE_NAME:
        raise RuntimeError(
            f"Tests must use database '{TEST_DATABASE_NAME}', "
            f"but configuration has '{settings.DATABASE_NAME}'"
        )
    if settings.MONGODB_URL != TEST_MONGODB_URL:
        raise RuntimeError(
            f"Tests must use MongoDB URL '{TEST_MONGODB_URL}', "
            f"but configuration has '{settings.MONGODB_URL}'"
        )
    if settings.ENVIRONMENT != "testing":
        raise RuntimeError(
            f"Tests must run in 'testing' environment, "
            f"but configuration has '{settings.ENVIRONMENT}'"
        )
    if DEV_DATABASE_NAME in settings.DATABASE_NAME and settings.DATABASE_NAME != TEST_DATABASE_NAME:
        raise RuntimeError(
            f"Tests must not use development database '{DEV_DATABASE_NAME}'. "
            f"Current database: '{settings.DATABASE_NAME}'"
        )


def verify_test_database_strict() -> None:
    """Strict safety check: database name must be exactly 'meditrust_test'."""
    if settings.DATABASE_NAME != TEST_DATABASE_NAME:
        raise RuntimeError(
            f"FAIL: Database name must be exactly '{TEST_DATABASE_NAME}', "
            f"got '{settings.DATABASE_NAME}'. "
            f"Tests are NOT allowed to connect to any other database."
        )


@pytest.fixture(scope="session", autouse=True)
def enforce_test_database():
    """Enforce test database isolation for the entire test session."""
    verify_test_database()
    verify_test_database_strict()
    yield
    verify_test_database()
    verify_test_database_strict()


@pytest.fixture(scope="session")
def test_db_manager() -> DatabaseManager:
    """Create a dedicated database manager for tests."""
    return DatabaseManager()


@pytest_asyncio.fixture(scope="function")
async def test_db(test_db_manager: DatabaseManager):
    """Connect to test database and clean ALL collections after each test."""
    await test_db_manager.connect()
    db = test_db_manager.db
    if db is not None:
        await _clean_all_collections(db)
    yield db
    if db is not None:
        await _clean_all_collections(db)
    await test_db_manager.disconnect()


async def _clean_all_collections(db) -> None:
    """Clean all user collections in the test database."""
    collection_names = await db.list_collection_names()
    for collection_name in collection_names:
        if not collection_name.startswith("system."):
            await db[collection_name].delete_many({})


@pytest.fixture(scope="session")
def app():
    """Create FastAPI application for testing with test settings."""
    return create_application()


@pytest_asyncio.fixture(scope="function")
async def client(app, test_db):
    """Create async test client with test database."""
    # Override the get_db dependency to use test database
    app.dependency_overrides[get_db_dep] = lambda: test_db
    
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    
    # Clean up dependency override
    app.dependency_overrides.clear()