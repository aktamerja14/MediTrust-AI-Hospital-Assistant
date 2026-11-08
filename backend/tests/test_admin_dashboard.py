"""Tests for Admin Dashboard API endpoints."""
import pytest
from httpx import AsyncClient
from bson import ObjectId
from datetime import timedelta

from app.main import create_application
from app.database.connection import db_manager
from app.services.auth import AuthService
from app.schemas.auth import UserCreate
from app.core.security.jwt import create_access_token
from tests.conftest import TEST_DATABASE_NAME, verify_test_database, DEV_DATABASE_NAME


def create_token(user_id: str, role: str, expires_delta: timedelta = timedelta(minutes=30)) -> str:
    """Create a JWT token for testing."""
    return create_access_token(subject=user_id, role=role, expires_delta=expires_delta)


class TestDatabaseIsolation:
    """Verify test database isolation from development database."""

    def test_database_name_is_test(self):
        """Test that DATABASE_NAME is exactly 'meditrust_test'."""
        from app.core.config import settings
        assert settings.DATABASE_NAME == TEST_DATABASE_NAME
        assert settings.DATABASE_NAME != "meditrust"
        assert settings.DATABASE_NAME != "meditrust_db"

    def test_environment_is_testing(self):
        """Test that ENVIRONMENT is 'testing'."""
        from app.core.config import settings
        assert settings.ENVIRONMENT == "testing"

    def test_verify_test_database_safety_check(self):
        """Test that the safety check function works correctly."""
        verify_test_database()

    def test_dev_database_not_used(self):
        """Test that development database name is not used in tests."""
        from app.core.config import settings
        assert "meditrust" not in settings.DATABASE_NAME or settings.DATABASE_NAME == TEST_DATABASE_NAME

    def test_dev_database_name_constant(self):
        """Test that development database name constant is correct."""
        assert DEV_DATABASE_NAME == "meditrust"

    def test_test_database_name_constant(self):
        """Test that test database name constant is correct."""
        assert TEST_DATABASE_NAME == "meditrust_test"

    def test_database_names_are_different(self):
        """Test that test and development database names are different."""
        assert TEST_DATABASE_NAME != DEV_DATABASE_NAME

    def test_mongodb_url_is_test_url(self):
        """Test that MONGODB_URL is the test URL."""
        from app.core.config import settings
        from tests.conftest import TEST_MONGODB_URL
        assert settings.MONGODB_URL == TEST_MONGODB_URL

    def test_strict_safety_check_passes(self):
        """Test that strict safety check passes with correct config."""
        from tests.conftest import verify_test_database_strict
        verify_test_database_strict()


class TestDatabaseCleaning:
    """Verify test database cleaning works correctly."""

    @pytest.mark.asyncio
    async def test_clean_all_collections_fixture(self, test_db):
        """Test that test_db fixture cleans all collections before and after tests."""
        # Insert test data into users collection
        await test_db.users.insert_one({"email": "test@example.com", "role": "patient"})
        
        # Verify data exists
        count = await test_db.users.count_documents({})
        assert count == 1
        
        # The fixture will clean after this test
        # Next test should start with clean database

    @pytest.mark.asyncio
    async def test_database_isolation_between_tests(self, test_db):
        """Test that each test runs with a clean database."""
        # This test runs after test_clean_all_collections_fixture
        # Database should be clean
        count = await test_db.users.count_documents({})
        assert count == 0


class TestAdminDashboard:
    """Test cases for GET /api/v1/admin/dashboard endpoint."""

    @pytest.mark.asyncio
    async def test_admin_with_valid_jwt_returns_200(self, client, test_db):
        """Test admin with valid JWT gets 200 with dashboard stats."""
        auth_service = AuthService(test_db)
        user_data = UserCreate(email="admin@test.com", password="adminPass123")
        admin_user = await auth_service.create_user(user_data, role="admin")

        token = create_token(admin_user.id, "admin")
        headers = {"Authorization": f"Bearer {token}"}

        response = await client.get("/api/v1/admin/dashboard", headers=headers)

        assert response.status_code == 200
        data = response.json()
        assert "total_users" in data
        assert "total_patients" in data
        assert "total_admins" in data
        assert "active_users" in data
        assert "inactive_users" in data
        assert isinstance(data["total_users"], int)
        assert data["total_users"] >= 1
        assert data["total_admins"] >= 1
        assert data["active_users"] >= 1

    @pytest.mark.asyncio
    async def test_patient_with_valid_jwt_returns_403(self, client, test_db):
        """Test patient with valid JWT gets 403 Forbidden."""
        auth_service = AuthService(test_db)
        user_data = UserCreate(email="patient@test.com", password="patientPass123")
        patient_user = await auth_service.create_user(user_data, role="patient")

        token = create_token(patient_user.id, "patient")
        headers = {"Authorization": f"Bearer {token}"}

        response = await client.get("/api/v1/admin/dashboard", headers=headers)

        assert response.status_code == 403
        data = response.json()
        assert "detail" in data
        assert "admin" in data["detail"].lower()

    @pytest.mark.asyncio
    async def test_missing_token_returns_401(self, client):
        """Test request without token returns 401 Unauthorized."""
        response = await client.get("/api/v1/admin/dashboard")

        assert response.status_code == 401
        data = response.json()
        assert "detail" in data

    @pytest.mark.asyncio
    async def test_invalid_token_returns_401(self, client):
        """Test request with invalid token returns 401 Unauthorized."""
        headers = {"Authorization": "Bearer invalid.token.here"}

        response = await client.get("/api/v1/admin/dashboard", headers=headers)

        assert response.status_code == 401
        data = response.json()
        assert "detail" in data

    @pytest.mark.asyncio
    async def test_expired_token_returns_401(self, client, test_db):
        """Test request with expired token returns 401 Unauthorized."""
        auth_service = AuthService(test_db)
        user_data = UserCreate(email="admin@test.com", password="adminPass123")
        admin_user = await auth_service.create_user(user_data, role="admin")

        expired_token = create_token(admin_user.id, "admin", expires_delta=timedelta(hours=-1))
        headers = {"Authorization": f"Bearer {expired_token}"}

        response = await client.get("/api/v1/admin/dashboard", headers=headers)

        assert response.status_code == 401
        data = response.json()
        assert "detail" in data

    @pytest.mark.asyncio
    async def test_inactive_admin_returns_403(self, client, test_db):
        """Test inactive admin user cannot access admin endpoints."""
        auth_service = AuthService(test_db)
        user_data = UserCreate(email="inactive_admin@test.com", password="inactivePass123")
        user = await auth_service.create_user(user_data, role="admin")
        await auth_service.collection.update_one(
            {"_id": ObjectId(user.id)},
            {"$set": {"is_active": False}}
        )
        user_doc = await auth_service.collection.find_one({"_id": ObjectId(user.id)})
        assert user_doc is not None
        updated_user_id = str(user_doc["_id"])

        token = create_token(updated_user_id, "admin")
        headers = {"Authorization": f"Bearer {token}"}

        response = await client.get("/api/v1/admin/dashboard", headers=headers)

        assert response.status_code in (401, 403)
        data = response.json()
        assert "detail" in data

    @pytest.mark.asyncio
    async def test_dashboard_counts_match_mongodb(self, client, test_db):
        """Test dashboard statistics match actual MongoDB counts."""
        auth_service = AuthService(test_db)
        
        admin_data = UserCreate(email="admin@test.com", password="adminPass123")
        admin_user = await auth_service.create_user(admin_data, role="admin")
        
        await auth_service.create_user(UserCreate(email="patient2@test.com", password="patientPass123"), role="patient")
        await auth_service.create_user(UserCreate(email="patient3@test.com", password="patientPass123"), role="patient")
        await auth_service.create_user(UserCreate(email="admin2@test.com", password="adminPass123"), role="admin")

        patients = await auth_service.collection.find({"role": "patient"}).to_list(length=10)
        if patients:
            await auth_service.collection.update_one(
                {"_id": patients[0]["_id"]},
                {"$set": {"is_active": False}}
            )

        token = create_token(admin_user.id, "admin")
        headers = {"Authorization": f"Bearer {token}"}

        response = await client.get("/api/v1/admin/dashboard", headers=headers)

        assert response.status_code == 200
        data = response.json()

        total_users = await auth_service.collection.count_documents({})
        total_patients = await auth_service.collection.count_documents({"role": "patient"})
        total_admins = await auth_service.collection.count_documents({"role": "admin"})
        active_users = await auth_service.collection.count_documents({"is_active": True})
        inactive_users = await auth_service.collection.count_documents({"is_active": False})

        assert data["total_users"] == total_users
        assert data["total_patients"] == total_patients
        assert data["total_admins"] == total_admins
        assert data["active_users"] == active_users
        assert data["inactive_users"] == inactive_users

    @pytest.mark.asyncio
    async def test_dashboard_no_password_hash_in_response(self, client, test_db):
        """Test that password hashes are not exposed in dashboard response."""
        auth_service = AuthService(test_db)
        user_data = UserCreate(email="admin@test.com", password="adminPass123")
        admin_user = await auth_service.create_user(user_data, role="admin")

        token = create_token(admin_user.id, "admin")
        headers = {"Authorization": f"Bearer {token}"}

        response = await client.get("/api/v1/admin/dashboard", headers=headers)

        assert response.status_code == 200
        data = response.json()
        response_text = str(data)
        assert "hashed_password" not in response_text
        assert "password" not in response_text


class TestExistingAuthEndpoints:
    """Verify existing authentication endpoints still work after admin changes."""

    @pytest.mark.asyncio
    async def test_patient_signup_works(self, client, test_db):
        """Test patient signup endpoint still works."""
        response = await client.post(
            "/api/v1/auth/signup",
            json={"email": "newpatient@test.com", "password": "newPass123"}
        )

        assert response.status_code == 201
        data = response.json()
        assert data["email"] == "newpatient@test.com"
        assert data["role"] == "patient"
        assert data["is_active"] is True
        assert "hashed_password" not in data
        assert "password" not in data

    @pytest.mark.asyncio
    async def test_patient_login_works(self, client, test_db):
        """Test patient login endpoint still works."""
        auth_service = AuthService(test_db)
        user_data = UserCreate(email="patient@test.com", password="patientPass123")
        await auth_service.create_user(user_data, role="patient")

        response = await client.post(
            "/api/v1/auth/login",
            json={"email": "patient@test.com", "password": "patientPass123"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert "expires_in" in data

    @pytest.mark.asyncio
    async def test_patient_me_endpoint_works(self, client, test_db):
        """Test /auth/me endpoint works for patients."""
        auth_service = AuthService(test_db)
        user_data = UserCreate(email="patient@test.com", password="patientPass123")
        patient_user = await auth_service.create_user(user_data, role="patient")

        token = create_token(patient_user.id, "patient")
        headers = {"Authorization": f"Bearer {token}"}

        response = await client.get("/api/v1/auth/me", headers=headers)

        assert response.status_code == 200
        data = response.json()
        assert data["email"] == "patient@test.com"
        assert data["role"] == "patient"

    @pytest.mark.asyncio
    async def test_admin_login_works(self, client, test_db):
        """Test admin login endpoint works."""
        auth_service = AuthService(test_db)
        user_data = UserCreate(email="admin@test.com", password="adminPass123")
        await auth_service.create_user(user_data, role="admin")

        response = await client.post(
            "/api/v1/auth/login",
            json={"email": "admin@test.com", "password": "adminPass123"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    @pytest.mark.asyncio
    async def test_admin_me_endpoint_works(self, client, test_db):
        """Test /auth/me endpoint works for admins."""
        auth_service = AuthService(test_db)
        user_data = UserCreate(email="admin@test.com", password="adminPass123")
        admin_user = await auth_service.create_user(user_data, role="admin")

        token = create_token(admin_user.id, "admin")
        headers = {"Authorization": f"Bearer {token}"}

        response = await client.get("/api/v1/auth/me", headers=headers)

        assert response.status_code == 200
        data = response.json()
        assert data["email"] == "admin@test.com"
        assert data["role"] == "admin"

    @pytest.mark.asyncio
    async def test_signup_never_assigns_admin_role(self, client, test_db):
        """Test public signup never assigns admin role."""
        response = await client.post(
            "/api/v1/auth/signup",
            json={"email": "attacker@test.com", "password": "attackPass123"}
        )

        assert response.status_code == 201
        data = response.json()
        assert data["role"] == "patient"
        assert data["role"] != "admin"