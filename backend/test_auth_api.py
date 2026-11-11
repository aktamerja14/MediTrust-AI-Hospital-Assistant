"""Complete Python test script for MediTrust Auth API endpoints.

This script tests the authentication endpoints directly using httpx with the FastAPI test client.
Run from the backend directory: python test_auth_api.py
"""
import asyncio
import os
import sys
from pathlib import Path

# Add backend to path
BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))

# Load .env before importing settings
from dotenv import load_dotenv
load_dotenv(BACKEND_DIR / ".env")

# Configure test environment
os.environ["ENVIRONMENT"] = "testing"
os.environ["DATABASE_NAME"] = "meditrust_test"
mongodb_uri = os.getenv("MONGODB_URL")
if not mongodb_uri:
    raise RuntimeError("MONGODB_URL environment variable is not set in backend/.env")
os.environ["MONGODB_URL"] = mongodb_uri

from app.main import create_application
from app.database.connection import DatabaseManager, get_db
from app.services.auth import AuthService
from app.schemas.auth import UserCreate, UserLogin
from app.api.dependencies import get_db as get_db_dep
from httpx import AsyncClient, ASGITransport


async def clean_test_db(db):
    """Clean all user collections in the test database."""
    collection_names = await db.list_collection_names()
    for collection_name in collection_names:
        if not collection_name.startswith("system."):
            await db[collection_name].delete_many({})


async def test_auth_endpoints():
    """Test all authentication API endpoints."""
    print("=" * 60)
    print("MediTrust Auth API Tests")
    print("=" * 60)

    # Create app and test database
    app = create_application()
    db_manager = DatabaseManager()
    await db_manager.connect()
    db = db_manager.db

    try:
        await clean_test_db(db)
    except Exception as e:
        print(f"Warning: Could not clean database: {e}")

    # Override get_db dependency
    app.dependency_overrides[get_db_dep] = lambda: db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        auth_service = AuthService(db)

        try:
            # Test 1: Patient Signup
            print("\n[TEST 1] Patient Signup - POST /api/v1/auth/signup")
            signup_payload = {
                "email": "testpatient@example.com",
                "password": "securePassword123"
            }
            response = await client.post("/api/v1/auth/signup", json=signup_payload)
            print(f"Status: {response.status_code}")
            print(f"Response: {response.json()}")
            assert response.status_code == 201, f"Signup failed: {response.text}"
            signup_data = response.json()
            assert signup_data["email"] == signup_payload["email"]
            assert signup_data["role"] == "patient"
            print("[OK] Signup successful")

            # Test 2: Duplicate Signup (should fail)
            print("\n[TEST 2] Duplicate Signup - Should return 400")
            response = await client.post("/api/v1/auth/signup", json=signup_payload)
            print(f"Status: {response.status_code}")
            print(f"Response: {response.json()}")
            assert response.status_code == 400
            assert "already registered" in response.json()["detail"].lower()
            print("[OK] Duplicate correctly rejected")

            # Test 3: Login with valid credentials
            print("\n[TEST 3] Login - POST /api/v1/auth/login")
            login_payload = {
                "email": "testpatient@example.com",
                "password": "securePassword123"
            }
            response = await client.post("/api/v1/auth/login", json=login_payload)
            print(f"Status: {response.status_code}")
            print(f"Response: {response.json()}")
            assert response.status_code == 200, f"Login failed: {response.text}"
            login_data = response.json()
            assert "access_token" in login_data
            assert login_data["token_type"] == "bearer"
            assert "expires_in" in login_data
            access_token = login_data["access_token"]
            print("[OK] Login successful, token received")

            # Test 4: Login with invalid password (should fail)
            print("\n[TEST 4] Login with invalid password - Should return 401")
            invalid_login = {
                "email": "testpatient@example.com",
                "password": "wrongPassword123"
            }
            response = await client.post("/api/v1/auth/login", json=invalid_login)
            print(f"Status: {response.status_code}")
            print(f"Response: {response.json()}")
            assert response.status_code == 401
            assert "invalid credentials" in response.json()["detail"].lower()
            print("[OK] Invalid credentials correctly rejected")

            # Test 5: Login with non-existent user (should fail with generic message)
            print("\n[TEST 5] Login with non-existent user - Should return 401")
            nonexistent_login = {
                "email": "nonexistent@example.com",
                "password": "somePassword123"
            }
            response = await client.post("/api/v1/auth/login", json=nonexistent_login)
            print(f"Status: {response.status_code}")
            print(f"Response: {response.json()}")
            assert response.status_code == 401
            assert "invalid credentials" in response.json()["detail"].lower()
            print("[OK] Non-existent user correctly rejected (no enumeration)")

            # Test 6: Get current user profile (/me) with valid token
            print("\n[TEST 6] Get Current User Profile - GET /api/v1/auth/me")
            headers = {"Authorization": f"Bearer {access_token}"}
            response = await client.get("/api/v1/auth/me", headers=headers)
            print(f"Status: {response.status_code}")
            print(f"Response: {response.json()}")
            assert response.status_code == 200, f"Get profile failed: {response.text}"
            profile_data = response.json()
            assert profile_data["email"] == signup_payload["email"]
            assert profile_data["role"] == "patient"
            assert profile_data["is_active"] is True
            print("[OK] Profile retrieved successfully")

            # Test 7: Get current user profile without token (should fail)
            print("\n[TEST 7] Get Profile without token - Should return 401")
            response = await client.get("/api/v1/auth/me")
            print(f"Status: {response.status_code}")
            print(f"Response: {response.json()}")
            assert response.status_code == 401
            print("[OK] Unauthorized correctly rejected")

            # Test 8: Get current user profile with invalid token (should fail)
            print("\n[TEST 8] Get Profile with invalid token - Should return 401")
            headers = {"Authorization": "Bearer invalid.token.here"}
            response = await client.get("/api/v1/auth/me", headers=headers)
            print(f"Status: {response.status_code}")
            print(f"Response: {response.json()}")
            assert response.status_code == 401
            print("[OK] Invalid token correctly rejected")

            # Test 9: Create admin user via service and test admin login
            print("\n[TEST 9] Admin User Creation and Login")
            admin_data = UserCreate(email="testadmin@example.com", password="adminPassword123")
            admin_user = await auth_service.create_user(admin_data, role="admin")
            print(f"Admin user created: {admin_user.email} (role: {admin_user.role})")

            admin_login = {
                "email": "testadmin@example.com",
                "password": "adminPassword123"
            }
            response = await client.post("/api/v1/auth/login", json=admin_login)
            print(f"Status: {response.status_code}")
            print(f"Response: {response.json()}")
            assert response.status_code == 200
            admin_token = response.json()["access_token"]
            print("[OK] Admin login successful")

            # Test 10: Admin accessing /me endpoint
            print("\n[TEST 10] Admin Get Profile")
            headers = {"Authorization": f"Bearer {admin_token}"}
            response = await client.get("/api/v1/auth/me", headers=headers)
            print(f"Status: {response.status_code}")
            print(f"Response: {response.json()}")
            assert response.status_code == 200
            admin_profile = response.json()
            assert admin_profile["email"] == "testadmin@example.com"
            assert admin_profile["role"] == "admin"
            print("[OK] Admin profile retrieved successfully")

            # Test 11: Validation errors - missing fields
            print("\n[TEST 11] Validation Error - Missing email")
            response = await client.post("/api/v1/auth/signup", json={"password": "password123"})
            print(f"Status: {response.status_code}")
            print(f"Response: {response.json()}")
            assert response.status_code == 422
            print("[OK] Validation error correctly returned")

            print("\n" + "=" * 60)
            print("ALL TESTS PASSED [OK]")
            print("=" * 60)

        finally:
            # Clean up
            app.dependency_overrides.clear()
            await clean_test_db(db)
            await db_manager.disconnect()


if __name__ == "__main__":
    asyncio.run(test_auth_endpoints())