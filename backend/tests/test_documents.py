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

# Valid TXT content
VALID_TXT_CONTENT = b"This is a valid plain text document for testing."

# Valid JSON content - array of objects
VALID_JSON_ARRAY_CONTENT = b'''[
  {"question": "What are visiting hours?", "answer": "9 AM to 6 PM"},
  {"question": "Where is emergency?", "answer": "Ground floor"}
]'''

# Valid JSON content - object
VALID_JSON_OBJECT_CONTENT = b'''{
  "hospital_name": "MediTrust",
  "departments": ["Cardiology", "Neurology"],
  "visiting_hours": "9 AM to 6 PM"
}'''

# Valid JSON content - nested
VALID_JSON_NESTED_CONTENT = b'''{
  "hospital": {
    "name": "MediTrust",
    "services": {
      "emergency": "24/7",
      "pharmacy": "Available"
    }
  }
}'''

# Valid JSON content - Gujarati/English
VALID_JSON_GUJARATI_CONTENT = '''{
  "hospital": "મેડીટ્રસ્ટ",
  "departments": ["કાર્ડિઓલૉજી", "ન્યુરોલૉજી"],
  "visiting_hours": "સવારે 9 થી સાંજ 6"
}'''.encode("utf-8")

# Invalid JSON content (malformed)
INVALID_JSON_CONTENT = b'{"key": "value",}'

# Empty JSON content
EMPTY_JSON_ARRAY_CONTENT = b"[]"
EMPTY_JSON_OBJECT_CONTENT = b"{}"

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
    async def test_admin_can_upload_valid_txt(self, client, test_db):
        """Test admin can upload a valid TXT file."""
        auth_service = AuthService(test_db)
        user_data = UserCreate(email="admin@test.com", password="adminPass123")
        admin_user = await auth_service.create_user(user_data, role="admin")

        token = create_token(admin_user.id, "admin")
        headers = {"Authorization": f"Bearer {token}"}

        files = {"file": ("test.txt", BytesIO(VALID_TXT_CONTENT), "text/plain")}
        response = await client.post("/api/v1/admin/documents/upload", files=files, headers=headers)

        assert response.status_code == 201
        data = response.json()
        assert "document_id" in data
        assert data["document_id"].startswith("doc_")
        assert data["original_filename"] == "test.txt"
        assert data["file_size"] == len(VALID_TXT_CONTENT)
        assert data["processing_status"] == "uploaded"
        assert "upload_timestamp" in data

    @pytest.mark.asyncio
    async def test_admin_can_upload_valid_json_array(self, client, test_db):
        """Test admin can upload a valid JSON array file."""
        auth_service = AuthService(test_db)
        user_data = UserCreate(email="admin@test.com", password="adminPass123")
        admin_user = await auth_service.create_user(user_data, role="admin")

        token = create_token(admin_user.id, "admin")
        headers = {"Authorization": f"Bearer {token}"}

        files = {"file": ("qa.json", BytesIO(VALID_JSON_ARRAY_CONTENT), "application/json")}
        response = await client.post("/api/v1/admin/documents/upload", files=files, headers=headers)

        assert response.status_code == 201
        data = response.json()
        assert "document_id" in data
        assert data["document_id"].startswith("doc_")
        assert data["original_filename"] == "qa.json"
        assert data["file_size"] == len(VALID_JSON_ARRAY_CONTENT)
        assert data["processing_status"] == "uploaded"
        assert "upload_timestamp" in data

    @pytest.mark.asyncio
    async def test_admin_can_upload_valid_json_object(self, client, test_db):
        """Test admin can upload a valid JSON object file."""
        auth_service = AuthService(test_db)
        user_data = UserCreate(email="admin@test.com", password="adminPass123")
        admin_user = await auth_service.create_user(user_data, role="admin")

        token = create_token(admin_user.id, "admin")
        headers = {"Authorization": f"Bearer {token}"}

        files = {"file": ("hospital.json", BytesIO(VALID_JSON_OBJECT_CONTENT), "application/json")}
        response = await client.post("/api/v1/admin/documents/upload", files=files, headers=headers)

        assert response.status_code == 201
        data = response.json()
        assert "document_id" in data
        assert data["document_id"].startswith("doc_")
        assert data["original_filename"] == "hospital.json"
        assert data["file_size"] == len(VALID_JSON_OBJECT_CONTENT)
        assert data["processing_status"] == "uploaded"
        assert "upload_timestamp" in data

    @pytest.mark.asyncio
    async def test_admin_can_upload_valid_json_nested(self, client, test_db):
        """Test admin can upload a valid nested JSON file."""
        auth_service = AuthService(test_db)
        user_data = UserCreate(email="admin@test.com", password="adminPass123")
        admin_user = await auth_service.create_user(user_data, role="admin")

        token = create_token(admin_user.id, "admin")
        headers = {"Authorization": f"Bearer {token}"}

        files = {"file": ("nested.json", BytesIO(VALID_JSON_NESTED_CONTENT), "application/json")}
        response = await client.post("/api/v1/admin/documents/upload", files=files, headers=headers)

        assert response.status_code == 201
        data = response.json()
        assert "document_id" in data
        assert data["document_id"].startswith("doc_")
        assert data["original_filename"] == "nested.json"
        assert data["file_size"] == len(VALID_JSON_NESTED_CONTENT)
        assert data["processing_status"] == "uploaded"
        assert "upload_timestamp" in data

    @pytest.mark.asyncio
    async def test_admin_can_upload_json_with_gujarati(self, client, test_db):
        """Test admin can upload JSON with Gujarati and English text."""
        auth_service = AuthService(test_db)
        user_data = UserCreate(email="admin@test.com", password="adminPass123")
        admin_user = await auth_service.create_user(user_data, role="admin")

        token = create_token(admin_user.id, "admin")
        headers = {"Authorization": f"Bearer {token}"}

        files = {"file": ("gujarati.json", BytesIO(VALID_JSON_GUJARATI_CONTENT), "application/json")}
        response = await client.post("/api/v1/admin/documents/upload", files=files, headers=headers)

        assert response.status_code == 201
        data = response.json()
        assert "document_id" in data
        assert data["document_id"].startswith("doc_")
        assert data["original_filename"] == "gujarati.json"
        assert data["file_size"] == len(VALID_JSON_GUJARATI_CONTENT)
        assert data["processing_status"] == "uploaded"
        assert "upload_timestamp" in data

    @pytest.mark.asyncio
    async def test_malformed_json_rejected(self, client, test_db):
        """Test malformed JSON is rejected."""
        auth_service = AuthService(test_db)
        user_data = UserCreate(email="admin@test.com", password="adminPass123")
        admin_user = await auth_service.create_user(user_data, role="admin")

        token = create_token(admin_user.id, "admin")
        headers = {"Authorization": f"Bearer {token}"}

        files = {"file": ("bad.json", BytesIO(INVALID_JSON_CONTENT), "application/json")}
        response = await client.post("/api/v1/admin/documents/upload", files=files, headers=headers)

        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert "json" in data["detail"].lower() or "syntax" in data["detail"].lower()

    @pytest.mark.asyncio
    async def test_empty_json_array_rejected(self, client, test_db):
        """Test empty JSON array is rejected."""
        auth_service = AuthService(test_db)
        user_data = UserCreate(email="admin@test.com", password="adminPass123")
        admin_user = await auth_service.create_user(user_data, role="admin")

        token = create_token(admin_user.id, "admin")
        headers = {"Authorization": f"Bearer {token}"}

        files = {"file": ("empty.json", BytesIO(EMPTY_JSON_ARRAY_CONTENT), "application/json")}
        response = await client.post("/api/v1/admin/documents/upload", files=files, headers=headers)

        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert "empty" in data["detail"].lower()

    @pytest.mark.asyncio
    async def test_empty_json_object_rejected(self, client, test_db):
        """Test empty JSON object is rejected."""
        auth_service = AuthService(test_db)
        user_data = UserCreate(email="admin@test.com", password="adminPass123")
        admin_user = await auth_service.create_user(user_data, role="admin")

        token = create_token(admin_user.id, "admin")
        headers = {"Authorization": f"Bearer {token}"}

        files = {"file": ("empty.json", BytesIO(EMPTY_JSON_OBJECT_CONTENT), "application/json")}
        response = await client.post("/api/v1/admin/documents/upload", files=files, headers=headers)

        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert "empty" in data["detail"].lower()

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

        files = {"file": ("test.exe", BytesIO(b"not a document"), "application/octet-stream")}
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
    async def test_txt_invalid_mime_type_rejected(self, client, test_db):
        """Test TXT file with invalid MIME type is rejected (validated as PDF due to MIME type)."""
        auth_service = AuthService(test_db)
        user_data = UserCreate(email="admin@test.com", password="adminPass123")
        admin_user = await auth_service.create_user(user_data, role="admin")

        token = create_token(admin_user.id, "admin")
        headers = {"Authorization": f"Bearer {token}"}

        # Upload .txt file but claim it's PDF - will fail PDF signature validation
        files = {"file": ("test.txt", BytesIO(VALID_TXT_CONTENT), "application/pdf")}
        response = await client.post("/api/v1/admin/documents/upload", files=files, headers=headers)

        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        # Since MIME type is application/pdf, it's validated as PDF and fails signature check
        assert "signature" in data["detail"].lower() or "pdf" in data["detail"].lower()

    @pytest.mark.asyncio
    async def test_empty_txt_rejected(self, client, test_db):
        """Test empty TXT file is rejected."""
        auth_service = AuthService(test_db)
        user_data = UserCreate(email="admin@test.com", password="adminPass123")
        admin_user = await auth_service.create_user(user_data, role="admin")

        token = create_token(admin_user.id, "admin")
        headers = {"Authorization": f"Bearer {token}"}

        files = {"file": ("empty.txt", BytesIO(EMPTY_PDF_CONTENT), "text/plain")}
        response = await client.post("/api/v1/admin/documents/upload", files=files, headers=headers)

        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert "empty" in data["detail"].lower()

    @pytest.mark.asyncio
    async def test_oversized_txt_rejected(self, client, test_db):
        """Test oversized TXT file is rejected."""
        auth_service = AuthService(test_db)
        user_data = UserCreate(email="admin@test.com", password="adminPass123")
        admin_user = await auth_service.create_user(user_data, role="admin")

        token = create_token(admin_user.id, "admin")
        headers = {"Authorization": f"Bearer {token}"}

        # Create content larger than 10MB
        oversized_content = b"x" * (11 * 1024 * 1024)
        files = {"file": ("large.txt", BytesIO(oversized_content), "text/plain")}
        response = await client.post("/api/v1/admin/documents/upload", files=files, headers=headers)

        assert response.status_code == 413
        data = response.json()
        assert "detail" in data

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


class TestJSONDocumentIngestion:
    """Tests for JSON document ingestion into RAG pipeline."""

    @pytest.mark.asyncio
    async def test_ingest_valid_json_array(self, client, test_db):
        """Test ingesting a valid JSON array document."""
        auth_service = AuthService(test_db)
        user_data = UserCreate(email="admin@test.com", password="adminPass123")
        admin_user = await auth_service.create_user(user_data, role="admin")

        token = create_token(admin_user.id, "admin")
        headers = {"Authorization": f"Bearer {token}"}

        # Upload JSON array
        files = {"file": ("qa.json", BytesIO(VALID_JSON_ARRAY_CONTENT), "application/json")}
        upload_response = await client.post("/api/v1/admin/documents/upload", files=files, headers=headers)
        assert upload_response.status_code == 201
        document_id = upload_response.json()["document_id"]

        # Ingest the document
        ingest_response = await client.post(
            f"/api/v1/admin/documents/{document_id}/ingest",
            headers=headers
        )
        assert ingest_response.status_code == 200
        data = ingest_response.json()
        assert data["status"] == "completed"
        assert data["chunks_created"] > 0
        assert data["document_id"] == document_id

        # Verify chunks stored in MongoDB
        from app.services.rag_ingestion import RAGIngestionService
        ingestion_service = RAGIngestionService(test_db)
        await ingestion_service.create_indexes()
        status = await ingestion_service.get_ingestion_status(document_id)
        assert status is not None
        assert status["chunks_count"] > 0
        assert status["processing_status"] == "completed"

    @pytest.mark.asyncio
    async def test_ingest_valid_json_object(self, client, test_db):
        """Test ingesting a valid JSON object document."""
        auth_service = AuthService(test_db)
        user_data = UserCreate(email="admin@test.com", password="adminPass123")
        admin_user = await auth_service.create_user(user_data, role="admin")

        token = create_token(admin_user.id, "admin")
        headers = {"Authorization": f"Bearer {token}"}

        # Upload JSON object
        files = {"file": ("hospital.json", BytesIO(VALID_JSON_OBJECT_CONTENT), "application/json")}
        upload_response = await client.post("/api/v1/admin/documents/upload", files=files, headers=headers)
        assert upload_response.status_code == 201
        document_id = upload_response.json()["document_id"]

        # Ingest the document
        ingest_response = await client.post(
            f"/api/v1/admin/documents/{document_id}/ingest",
            headers=headers
        )
        assert ingest_response.status_code == 200
        data = ingest_response.json()
        assert data["status"] == "completed"
        assert data["chunks_created"] > 0

    @pytest.mark.asyncio
    async def test_ingest_json_with_gujarati(self, client, test_db):
        """Test ingesting JSON with Gujarati text."""
        auth_service = AuthService(test_db)
        user_data = UserCreate(email="admin@test.com", password="adminPass123")
        admin_user = await auth_service.create_user(user_data, role="admin")

        token = create_token(admin_user.id, "admin")
        headers = {"Authorization": f"Bearer {token}"}

        # Upload JSON with Gujarati
        files = {"file": ("gujarati.json", BytesIO(VALID_JSON_GUJARATI_CONTENT), "application/json")}
        upload_response = await client.post("/api/v1/admin/documents/upload", files=files, headers=headers)
        assert upload_response.status_code == 201
        document_id = upload_response.json()["document_id"]

        # Ingest the document
        ingest_response = await client.post(
            f"/api/v1/admin/documents/{document_id}/ingest",
            headers=headers
        )
        assert ingest_response.status_code == 200
        data = ingest_response.json()
        assert data["status"] == "completed"
        assert data["chunks_created"] > 0


class TestFullWorkflow:
    """Tests for complete upload-to-chatbot workflow with JSON documents."""

    @pytest.mark.asyncio
    async def test_json_upload_ingest_chat_workflow(self, client, test_db):
        """Test complete workflow: upload JSON -> ingest -> verify chunks -> chat works."""
        auth_service = AuthService(test_db)
        user_data = UserCreate(email="admin@test.com", password="adminPass123")
        admin_user = await auth_service.create_user(user_data, role="admin")

        token = create_token(admin_user.id, "admin")
        headers = {"Authorization": f"Bearer {token}"}

        # JSON content with specific Q&A
        json_content = b'''[
          {"question": "What are MediTrust visiting hours?", "answer": "MediTrust visiting hours are 9 AM to 6 PM daily"},
          {"question": "Where is the emergency department?", "answer": "Emergency department is on the ground floor, available 24/7"},
          {"question": "Does MediTrust have a pharmacy?", "answer": "Yes, MediTrust has an in-house pharmacy available during hospital hours"}
        ]'''

        # 1. Upload JSON document
        files = {"file": ("faq.json", BytesIO(json_content), "application/json")}
        upload_response = await client.post("/api/v1/admin/documents/upload", files=files, headers=headers)
        assert upload_response.status_code == 201
        document_id = upload_response.json()["document_id"]

        # 2. Ingest the document
        ingest_response = await client.post(
            f"/api/v1/admin/documents/{document_id}/ingest",
            headers=headers
        )
        assert ingest_response.status_code == 200
        data = ingest_response.json()
        assert data["status"] == "completed"
        assert data["chunks_created"] > 0

        # 3. Verify chunks are stored in vector database (MongoDB)
        from app.services.rag_ingestion import RAGIngestionService
        ingestion_service = RAGIngestionService(test_db)
        await ingestion_service.create_indexes()
        status = await ingestion_service.get_ingestion_status(document_id)
        assert status is not None
        assert status["chunks_count"] > 0
        assert status["processing_status"] == "completed"

        # 4. Chat endpoint works (uses mock LLM in tests)
        patient_user_data = UserCreate(email="patient@test.com", password="patientPass123")
        patient_user = await auth_service.create_user(patient_user_data, role="patient")
        patient_token = create_token(patient_user.id, "patient")
        patient_headers = {"Authorization": f"Bearer {patient_token}"}

        chat_response = await client.post("/api/v1/chat/message", json={
            "message": "Hello, what are the visiting hours?"
        }, headers=patient_headers)

        assert chat_response.status_code == 200
        chat_data = chat_response.json()
        assert "assistant_message" in chat_data
        assert "content" in chat_data["assistant_message"]
        assert len(chat_data["assistant_message"]["content"]) > 0

        # 5. Verify existing PDF/TXT upload still works
        files = {"file": ("test.pdf", BytesIO(VALID_PDF_CONTENT), "application/pdf")}
        upload_response2 = await client.post("/api/v1/admin/documents/upload", files=files, headers=headers)
        assert upload_response2.status_code == 201


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