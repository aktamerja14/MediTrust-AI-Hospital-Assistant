"""Tests for Admin Document Management API endpoints."""
import os
import tempfile
import pytest
from pathlib import Path
from datetime import timedelta
from httpx import AsyncClient
from io import BytesIO

from app.main import create_application
from app.database.connection import db_manager
from app.services.auth import AuthService
from app.schemas.auth import UserCreate
from app.core.security.jwt import create_access_token
from tests.conftest import TEST_DATABASE_NAME, verify_test_database


def create_token(user_id: str, role: str, expires_delta: timedelta = timedelta(minutes=30)) -> str:
    """Create a JWT token for testing."""
    return create_access_token(subject=user_id, role=role, expires_delta=expires_delta)


# Valid PDF content (minimal valid PDF)
VALID_PDF_CONTENT = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>
endobj
xref
0 4
0000000000 65535 f
0000000010 00000 n
0000000060 00000 n
0000000117 00000 n
trailer
<< /Size 4 /Root 1 0 R >>
startxref
174
%%EOF
"""

# Invalid PDF content (wrong signature)
INVALID_PDF_CONTENT = b"This is not a PDF file"

# Empty PDF content
EMPTY_PDF_CONTENT = b""


class TestDocumentUpload:
    """Tests for POST /api/v1/admin/documents/upload endpoint."""

    @pytest.mark.asyncio
    async def test_admin_can_upload_valid_pdf(self, client, test_db):
        """Test admin can upload a valid PDF file."""
        auth_service = AuthService(test_db)
        user_data = UserCreate(email="admin@test.com", password="adminPass123")
        admin_user = await auth_service.create_user(user_data, role="admin")

        token = create_token(admin_user.id, "admin")
        headers = {"Authorization": f"Bearer {token}"}

        files = {"file": ("test.pdf", BytesIO(VALID_PDF_CONTENT), "application/pdf")}
        response = await client.post("/api/v1/admin/documents/upload", files=files, headers=headers)

        assert response.status_code == 201
        data = response.json()
        assert "document_id" in data
        assert data["document_id"].startswith("doc_")
        assert data["original_filename"] == "test.pdf"
        assert data["file_size"] == len(VALID_PDF_CONTENT)
        assert data["processing_status"] == "uploaded"
        assert "upload_timestamp" in data

    @pytest.mark.asyncio
    async def test_patient_upload_returns_403(self, client, test_db):
        """Test patient upload returns 403 Forbidden."""
        auth_service = AuthService(test_db)
        user_data = UserCreate(email="patient@test.com", password="patientPass123")
        patient_user = await auth_service.create_user(user_data, role="patient")

        token = create_token(patient_user.id, "patient")
        headers = {"Authorization": f"Bearer {token}"}

        files = {"file": ("test.pdf", BytesIO(VALID_PDF_CONTENT), "application/pdf")}
        response = await client.post("/api/v1/admin/documents/upload", files=files, headers=headers)

        assert response.status_code == 403
        data = response.json()
        assert "detail" in data
        assert "admin" in data["detail"].lower()

    @pytest.mark.asyncio
    async def test_missing_token_returns_401(self, client):
        """Test upload without token returns 401 Unauthorized."""
        files = {"file": ("test.pdf", BytesIO(VALID_PDF_CONTENT), "application/pdf")}
        response = await client.post("/api/v1/admin/documents/upload", files=files)

        assert response.status_code == 401
        data = response.json()
        assert "detail" in data

    @pytest.mark.asyncio
    async def test_invalid_token_returns_401(self, client):
        """Test upload with invalid token returns 401 Unauthorized."""
        headers = {"Authorization": "Bearer invalid.token.here"}
        files = {"file": ("test.pdf", BytesIO(VALID_PDF_CONTENT), "application/pdf")}
        response = await client.post("/api/v1/admin/documents/upload", files=files, headers=headers)

        assert response.status_code == 401
        data = response.json()
        assert "detail" in data

    @pytest.mark.asyncio
    async def test_invalid_extension_rejected(self, client, test_db):
        """Test file with invalid extension is rejected."""
        auth_service = AuthService(test_db)
        user_data = UserCreate(email="admin@test.com", password="adminPass123")
        admin_user = await auth_service.create_user(user_data, role="admin")

        token = create_token(admin_user.id, "admin")
        headers = {"Authorization": f"Bearer {token}"}

        files = {"file": ("test.txt", BytesIO(b"not a pdf"), "text/plain")}
        response = await client.post("/api/v1/admin/documents/upload", files=files, headers=headers)

        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert "extension" in data["detail"].lower()

    @pytest.mark.asyncio
    async def test_invalid_mime_type_rejected(self, client, test_db):
        """Test file with invalid MIME type is rejected."""
        auth_service = AuthService(test_db)
        user_data = UserCreate(email="admin@test.com", password="adminPass123")
        admin_user = await auth_service.create_user(user_data, role="admin")

        token = create_token(admin_user.id, "admin")
        headers = {"Authorization": f"Bearer {token}"}

        files = {"file": ("test.pdf", BytesIO(VALID_PDF_CONTENT), "image/png")}
        response = await client.post("/api/v1/admin/documents/upload", files=files, headers=headers)

        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert "mime" in data["detail"].lower()

    @pytest.mark.asyncio
    async def test_fake_pdf_signature_rejected(self, client, test_db):
        """Test file with incorrect PDF signature is rejected."""
        auth_service = AuthService(test_db)
        user_data = UserCreate(email="admin@test.com", password="adminPass123")
        admin_user = await auth_service.create_user(user_data, role="admin")

        token = create_token(admin_user.id, "admin")
        headers = {"Authorization": f"Bearer {token}"}

        files = {"file": ("test.pdf", BytesIO(INVALID_PDF_CONTENT), "application/pdf")}
        response = await client.post("/api/v1/admin/documents/upload", files=files, headers=headers)

        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert "signature" in data["detail"].lower() or "pdf" in data["detail"].lower()

    @pytest.mark.asyncio
    async def test_empty_pdf_rejected(self, client, test_db):
        """Test empty PDF file is rejected."""
        auth_service = AuthService(test_db)
        user_data = UserCreate(email="admin@test.com", password="adminPass123")
        admin_user = await auth_service.create_user(user_data, role="admin")

        token = create_token(admin_user.id, "admin")
        headers = {"Authorization": f"Bearer {token}"}

        files = {"file": ("empty.pdf", BytesIO(EMPTY_PDF_CONTENT), "application/pdf")}
        response = await client.post("/api/v1/admin/documents/upload", files=files, headers=headers)

        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert "empty" in data["detail"].lower()

    @pytest.mark.asyncio
    async def test_oversized_pdf_rejected(self, client, test_db):
        """Test oversized PDF file is rejected."""
        auth_service = AuthService(test_db)
        user_data = UserCreate(email="admin@test.com", password="adminPass123")
        admin_user = await auth_service.create_user(user_data, role="admin")

        token = create_token(admin_user.id, "admin")
        headers = {"Authorization": f"Bearer {token}"}

        # Create content larger than 10MB
        oversized_content = b"x" * (11 * 1024 * 1024)
        files = {"file": ("large.pdf", BytesIO(oversized_content), "application/pdf")}
        response = await client.post("/api/v1/admin/documents/upload", files=files, headers=headers)

        assert response.status_code == 413
        data = response.json()
        assert "detail" in data


class TestDocumentList:
    """Tests for GET /api/v1/admin/documents endpoint."""

    @pytest.mark.asyncio
    async def test_admin_can_list_documents(self, client, test_db):
        """Test admin can list uploaded documents."""
        auth_service = AuthService(test_db)
        user_data = UserCreate(email="admin@test.com", password="adminPass123")
        admin_user = await auth_service.create_user(user_data, role="admin")

        token = create_token(admin_user.id, "admin")
        headers = {"Authorization": f"Bearer {token}"}

        # Upload a document first
        files = {"file": ("test1.pdf", BytesIO(VALID_PDF_CONTENT), "application/pdf")}
        await client.post("/api/v1/admin/documents/upload", files=files, headers=headers)

        # List documents
        response = await client.get("/api/v1/admin/documents", headers=headers)

        assert response.status_code == 200
        data = response.json()
        assert "documents" in data
        assert "total" in data
        assert "page" in data
        assert "page_size" in data
        assert "total_pages" in data
        assert data["total"] >= 1
        assert len(data["documents"]) >= 1
        doc = data["documents"][0]
        assert "document_id" in doc
        assert "original_filename" in doc
        assert "file_type" in doc
        assert "file_size" in doc
        assert "upload_timestamp" in doc
        assert "processing_status" in doc
        assert "uploaded_by" in doc
        # Should not expose internal storage paths
        assert "storage_filename" in doc  # This is metadata, not full path

    @pytest.mark.asyncio
    async def test_pagination_works(self, client, test_db):
        """Test pagination works correctly."""
        auth_service = AuthService(test_db)
        user_data = UserCreate(email="admin@test.com", password="adminPass123")
        admin_user = await auth_service.create_user(user_data, role="admin")

        token = create_token(admin_user.id, "admin")
        headers = {"Authorization": f"Bearer {token}"}

        # Upload multiple documents
        for i in range(3):
            files = {"file": (f"test{i}.pdf", BytesIO(VALID_PDF_CONTENT), "application/pdf")}
            await client.post("/api/v1/admin/documents/upload", files=files, headers=headers)

        # Test page 1 with page_size=2
        response = await client.get("/api/v1/admin/documents?page=1&page_size=2", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert data["page"] == 1
        assert data["page_size"] == 2
        assert len(data["documents"]) == 2
        assert data["total"] >= 3
        assert data["total_pages"] >= 2

        # Test page 2 with page_size=2
        response = await client.get("/api/v1/admin/documents?page=2&page_size=2", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert data["page"] == 2
        assert data["page_size"] == 2
        assert len(data["documents"]) >= 1


class TestDocumentDelete:
    """Tests for DELETE /api/v1/admin/documents/{document_id} endpoint."""

    @pytest.mark.asyncio
    async def test_admin_can_delete_document(self, client, test_db):
        """Test admin can delete a document."""
        auth_service = AuthService(test_db)
        user_data = UserCreate(email="admin@test.com", password="adminPass123")
        admin_user = await auth_service.create_user(user_data, role="admin")

        token = create_token(admin_user.id, "admin")
        headers = {"Authorization": f"Bearer {token}"}

        # Upload a document first
        files = {"file": ("test.pdf", BytesIO(VALID_PDF_CONTENT), "application/pdf")}
        upload_response = await client.post("/api/v1/admin/documents/upload", files=files, headers=headers)
        document_id = upload_response.json()["document_id"]

        # Delete the document
        response = await client.delete(f"/api/v1/admin/documents/{document_id}", headers=headers)

        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Document deleted successfully"
        assert data["document_id"] == document_id

        # Verify document is gone
        list_response = await client.get("/api/v1/admin/documents", headers=headers)
        assert list_response.json()["total"] == 0

    @pytest.mark.asyncio
    async def test_deleted_document_file_removed(self, client, test_db):
        """Test that deleted document's stored file is removed from disk."""
        auth_service = AuthService(test_db)
        user_data = UserCreate(email="admin@test.com", password="adminPass123")
        admin_user = await auth_service.create_user(user_data, role="admin")

        token = create_token(admin_user.id, "admin")
        headers = {"Authorization": f"Bearer {token}"}

        # Upload a document first
        files = {"file": ("test.pdf", BytesIO(VALID_PDF_CONTENT), "application/pdf")}
        upload_response = await client.post("/api/v1/admin/documents/upload", files=files, headers=headers)
        document_id = upload_response.json()["document_id"]

        # Get the document metadata to find storage filename
        list_response = await client.get("/api/v1/admin/documents", headers=headers)
        doc = list_response.json()["documents"][0]
        storage_filename = doc["storage_filename"]

        # Verify file exists before deletion
        from app.core.config import settings
        storage_path = Path(settings.DOCUMENT_STORAGE_PATH).resolve() / storage_filename
        assert storage_path.exists(), "File should exist before deletion"

        # Delete the document
        await client.delete(f"/api/v1/admin/documents/{document_id}", headers=headers)

        # Verify file is removed
        assert not storage_path.exists(), "File should be deleted after document deletion"

    @pytest.mark.asyncio
    async def test_missing_document_returns_404(self, client, test_db):
        """Test deleting non-existent document returns 404."""
        auth_service = AuthService(test_db)
        user_data = UserCreate(email="admin@test.com", password="adminPass123")
        admin_user = await auth_service.create_user(user_data, role="admin")

        token = create_token(admin_user.id, "admin")
        headers = {"Authorization": f"Bearer {token}"}

        response = await client.delete("/api/v1/admin/documents/doc_nonexistent123", headers=headers)

        assert response.status_code == 404
        data = response.json()
        assert "detail" in data

    @pytest.mark.asyncio
    async def test_path_traversal_rejected(self, client, test_db):
        """Test path traversal attempts are rejected."""
        auth_service = AuthService(test_db)
        user_data = UserCreate(email="admin@test.com", password="adminPass123")
        admin_user = await auth_service.create_user(user_data, role="admin")

        token = create_token(admin_user.id, "admin")
        headers = {"Authorization": f"Bearer {token}"}

        # Try to delete with path traversal in document_id
        response = await client.delete("/api/v1/admin/documents/../../etc/passwd", headers=headers)

        # Should return 404 (not found) or 400 (bad request), not 200
        assert response.status_code in (400, 404)
        data = response.json()
        assert "detail" in data


class TestExistingAuthEndpointsStillWork:
    """Verify existing authentication endpoints still work after document changes."""

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

    @pytest.mark.asyncio
    async def test_admin_dashboard_still_works(self, client, test_db):
        """Test admin dashboard endpoint still works."""
        auth_service = AuthService(test_db)
        user_data = UserCreate(email="admin@test.com", password="adminPass123")
        admin_user = await auth_service.create_user(user_data, role="admin")

        token = create_token(admin_user.id, "admin")
        headers = {"Authorization": f"Bearer {token}"}

        response = await client.get("/api/v1/admin/dashboard", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert "total_users" in data