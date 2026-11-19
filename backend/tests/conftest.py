"""Test configuration and fixtures for MediTrust backend tests."""
import os
import pytest
import pytest_asyncio
from pathlib import Path
from typing import AsyncGenerator
from dotenv import load_dotenv

# Load backend/.env before reading environment variables
BACKEND_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BACKEND_DIR / ".env")

# Fail immediately if MONGODB_URL is missing or empty
mongodb_uri = os.getenv("MONGODB_URL")
if not mongodb_uri:
    raise RuntimeError(
        "MONGODB_URL environment variable is not set. "
        "Tests require a valid MongoDB connection URI."
    )

# Configure test environment variables BEFORE importing application settings
os.environ["ENVIRONMENT"] = "testing"
os.environ["DATABASE_NAME"] = "meditrust_test"
os.environ["MONGODB_URL"] = mongodb_uri
os.environ["LLM_PROVIDER"] = "mock"
os.environ["LLM_MODEL"] = "mock-model"
os.environ["LLM_API_KEY"] = "mock-key"

from app.core.config import settings
from app.database.connection import DatabaseManager, get_db
from app.main import create_application
from app.api.dependencies import get_db as get_db_dep
from httpx import AsyncClient, ASGITransport


TEST_DATABASE_NAME = "meditrust_test"
TEST_MONGODB_URL = mongodb_uri
DEV_DATABASE_NAME = "meditrust"


def verify_test_database() -> None:
    """Safety check: ensure tests only run against the test database."""
    if settings.DATABASE_NAME != TEST_DATABASE_NAME:
        raise RuntimeError(
            f"Tests must use database '{TEST_DATABASE_NAME}', "
            f"but configuration has '{settings.DATABASE_NAME}'"
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
        # Populate doctors for tests
        await _populate_doctors(db)
    yield db
    if db is not None:
        await _clean_all_collections(db)
    await test_db_manager.disconnect()


async def _populate_doctors(db) -> None:
    """Populate doctors collection with test data."""
    doctors_data = [
        {
            "doctor_id": "dr.deven",
            "name": "Dr. Deven Patel",
            "qualifications": "MBBS",
            "specialization": "General Physician (સામાન્ય રોગોની સારવાર)",
            "experience_years": 6,
            "photo_path": "doctor_photos/dr_deven.svg",
            "phone": "9700112232",
            "available_from": "10 a.m.",
            "available_to": "7:30 p.m.",
        },
        {
            "doctor_id": "dr.rahul",
            "name": "Dr. Rahul Maheta",
            "qualifications": "MBBS, MD (Neurology)",
            "specialization": "Neurologist (મગજ અને નર્વસ સિસ્ટમ)",
            "experience_years": 5,
            "photo_path": "doctor_photos/dr_rahul.svg",
            "phone": "9900112233",
            "available_from": "9 a.m.",
            "available_to": "6 p.m.",
        },
        {
            "doctor_id": "dr.niddhi",
            "name": "Dr. Nidhi Gopani",
            "qualifications": "MBBS, MD (Dermatology)",
            "specialization": "Dermatologist (ત્વચા, વાળ, નખ)",
            "experience_years": 7,
            "photo_path": "doctor_photos/dr_niddhi.svg",
            "phone": "9812345678",
            "available_from": "11 a.m.",
            "available_to": "7 p.m.",
        },
        {
            "doctor_id": "dr.nita",
            "name": "Dr. Nita Surani",
            "qualifications": "MBBS, MD (Orthopedics)",
            "specialization": "Orthopedics (હાડકાં અને સાંધાના રોગ)",
            "experience_years": 5,
            "photo_path": "doctor_photos/dr_neeta.svg",
            "phone": "9898123456",
            "available_from": "9:30 a.m.",
            "available_to": "5:30 p.m.",
        },
        {
            "doctor_id": "dr.suresh",
            "name": "Dr. Suresh Patel",
            "qualifications": "MBBS, MS (Gynecology)",
            "specialization": "Gynecologist (મહિલાઓના રોગ / ગર્ભાવસ્થા)",
            "experience_years": 8,
            "photo_path": "doctor_photos/dr_shuresh.svg",
            "phone": "9823456712",
            "available_from": "5 a.m.",
            "available_to": "10 p.m.",
        },
        {
            "doctor_id": "dr.mahesh",
            "name": "Dr. Mahesh Ramani",
            "qualifications": "MBBS, MD (Cardiology)",
            "specialization": "Cardiologist (હૃદયના રોગ)",
            "experience_years": 12,
            "photo_path": "doctor_photos/dr_mahesh_GyexvM8.svg",
            "phone": "9876543210",
            "available_from": "5 a.m.",
            "available_to": "11 p.m.",
        },
    ]

    for doctor in doctors_data:
        await db["doctors"].update_one(
            {"doctor_id": doctor["doctor_id"]},
            {"$set": doctor},
            upsert=True
        )

    await db["doctors"].create_index("doctor_id", unique=True)


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


@pytest_asyncio.fixture(scope="function")
async def auth_headers(client: AsyncClient, test_db):
    """Create authenticated user and return auth headers."""
    from app.core.security.password import hash_password
    from app.core.security.jwt import create_access_token

    # Create test user
    test_email = "test_patient@example.com"
    test_password = "testpassword123"
    hashed = hash_password(test_password)

    user_doc = {
        "email": test_email,
        "hashed_password": hashed,
        "role": "patient",
        "is_active": True,
    }
    result = await test_db["users"].insert_one(user_doc)
    user_id = str(result.inserted_id)

    # Create access token
    token = create_access_token(
        subject=user_id,
        role="patient",
        expires_delta=None
    )

    return {"Authorization": f"Bearer {token}"}
