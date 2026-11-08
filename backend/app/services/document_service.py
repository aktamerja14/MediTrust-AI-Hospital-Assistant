"""Document management business logic service."""
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from pymongo.asynchronous.database import AsyncDatabase
from pymongo.errors import PyMongoError

from app.core.config import settings

logger = logging.getLogger("meditrust.services.document")

DOCUMENTS_COLLECTION = "documents"
PDF_SIGNATURE = b"%PDF"
ALLOWED_MIME_TYPES = {"application/pdf"}
ALLOWED_EXTENSIONS = {".pdf"}


class DocumentServiceError(Exception):
    """Base exception for document service errors."""
    pass


class InvalidFileError(DocumentServiceError):
    """Raised when uploaded file is invalid."""
    pass


class FileSizeError(DocumentServiceError):
    """Raised when file exceeds maximum size."""
    pass


class StorageError(DocumentServiceError):
    """Raised when file storage operation fails."""
    pass


class DocumentNotFoundError(DocumentServiceError):
    """Raised when document is not found."""
    pass


class DocumentService:
    """Service layer for document management operations."""

    def __init__(self, db: AsyncDatabase) -> None:
        self.db = db
        self.collection = db[DOCUMENTS_COLLECTION]
        self.storage_path = Path(settings.DOCUMENT_STORAGE_PATH).resolve()
        self._ensure_storage_directory()

    def _ensure_storage_directory(self) -> None:
        """Create storage directory if it doesn't exist."""
        try:
            self.storage_path.mkdir(parents=True, exist_ok=True)
            logger.info("Document storage directory ensured at %s", self.storage_path)
        except OSError as exc:
            logger.error("Failed to create storage directory: %s", exc)
            raise StorageError("Unable to initialize document storage") from exc

    def _validate_file_extension(self, filename: str) -> None:
        """Validate file extension."""
        ext = Path(filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise InvalidFileError(
                f"Invalid file extension '{ext}'. Only .pdf files are allowed."
            )

    def _validate_mime_type(self, content_type: Optional[str]) -> None:
        """Validate MIME type."""
        if not content_type or content_type not in ALLOWED_MIME_TYPES:
            raise InvalidFileError(
                f"Invalid MIME type '{content_type}'. Only application/pdf is allowed."
            )

    def _validate_file_size(self, file_size: int) -> None:
        """Validate file size."""
        if file_size == 0:
            raise InvalidFileError("Empty files are not allowed")
        if file_size > settings.MAX_UPLOAD_SIZE:
            max_mb = settings.MAX_UPLOAD_SIZE / (1024 * 1024)
            raise FileSizeError(
                f"File size {file_size} bytes exceeds maximum allowed size of {max_mb} MB"
            )

    def _validate_pdf_signature(self, file_content: bytes) -> None:
        """Validate PDF file signature (magic bytes)."""
        if len(file_content) < 4:
            raise InvalidFileError("File too small to be a valid PDF")
        if not file_content.startswith(PDF_SIGNATURE):
            raise InvalidFileError("Invalid PDF file signature")

    def _validate_pdf_structure(self, file_content: bytes) -> None:
        """Basic PDF structure validation."""
        try:
            content_str = file_content.decode("latin-1", errors="ignore")
            if "%%EOF" not in content_str:
                raise InvalidFileError("Invalid PDF structure: missing EOF marker")
        except Exception as exc:
            raise InvalidFileError("PDF structure validation failed") from exc

    def _generate_document_id(self) -> str:
        """Generate unique document ID."""
        return f"doc_{uuid.uuid4().hex[:24]}"

    def _generate_storage_filename(self, document_id: str, original_filename: str) -> str:
        """Generate secure storage filename."""
        ext = Path(original_filename).suffix.lower()
        return f"{document_id}{ext}"

    def _get_storage_path(self, storage_filename: str) -> Path:
        """Get full storage path with path traversal protection."""
        storage_file = self.storage_path / storage_filename
        try:
            storage_file.resolve().relative_to(self.storage_path)
        except ValueError:
            raise StorageError("Invalid storage filename: path traversal attempt detected")
        return storage_file

    async def _store_file(self, storage_filename: str, content: bytes) -> Path:
        """Store file content to disk."""
        storage_file = self._get_storage_path(storage_filename)
        try:
            storage_file.write_bytes(content)
            logger.info("Stored document file: %s", storage_filename)
            return storage_file
        except OSError as exc:
            logger.error("Failed to store document file: %s", exc)
            raise StorageError("Failed to save document file") from exc

    async def _delete_file(self, storage_filename: str) -> bool:
        """Delete file from disk. Returns True if file was deleted."""
        storage_file = self._get_storage_path(storage_filename)
        try:
            if storage_file.exists():
                storage_file.unlink()
                logger.info("Deleted document file: %s", storage_filename)
                return True
            logger.warning("Document file not found for deletion: %s", storage_filename)
            return False
        except OSError as exc:
            logger.error("Failed to delete document file: %s", exc)
            raise StorageError("Failed to delete document file") from exc

    async def _create_metadata(
        self,
        document_id: str,
        original_filename: str,
        storage_filename: str,
        file_type: str,
        file_size: int,
        uploaded_by: str,
    ) -> dict:
        """Create document metadata document."""
        metadata = {
            "document_id": document_id,
            "original_filename": original_filename,
            "storage_filename": storage_filename,
            "file_type": file_type,
            "file_size": file_size,
            "upload_timestamp": datetime.utcnow(),
            "processing_status": "uploaded",
            "uploaded_by": uploaded_by,
        }
        return metadata

    async def _insert_metadata(self, metadata: dict) -> None:
        """Insert document metadata into MongoDB."""
        try:
            await self.collection.insert_one(metadata)
            logger.info("Inserted document metadata: %s", metadata["document_id"])
        except PyMongoError as exc:
            logger.error("Failed to insert document metadata: %s", exc)
            raise DocumentServiceError("Failed to save document metadata") from exc

    async def _get_metadata(self, document_id: str) -> Optional[dict]:
        """Get document metadata from MongoDB."""
        try:
            return await self.collection.find_one({"document_id": document_id})
        except PyMongoError as exc:
            logger.error("Failed to retrieve document metadata: %s", exc)
            raise DocumentServiceError("Failed to retrieve document metadata") from exc

    async def _delete_metadata(self, document_id: str) -> bool:
        """Delete document metadata from MongoDB. Returns True if deleted."""
        try:
            result = await self.collection.delete_one({"document_id": document_id})
            if result.deleted_count > 0:
                logger.info("Deleted document metadata: %s", document_id)
                return True
            logger.warning("Document metadata not found for deletion: %s", document_id)
            return False
        except PyMongoError as exc:
            logger.error("Failed to delete document metadata: %s", exc)
            raise DocumentServiceError("Failed to delete document metadata") from exc

    async def upload_document(
        self,
        file_content: bytes,
        filename: str,
        content_type: Optional[str],
        uploaded_by: str,
    ) -> dict:
        """
        Upload and validate a PDF document.

        Args:
            file_content: Raw file bytes
            filename: Original filename
            content_type: MIME type from upload
            uploaded_by: User ID of uploader

        Returns:
            Document metadata dict

        Raises:
            InvalidFileError: If file validation fails
            FileSizeError: If file exceeds size limit
            StorageError: If file storage fails
            DocumentServiceError: If metadata save fails
        """
        self._validate_file_extension(filename)
        self._validate_mime_type(content_type)
        self._validate_file_size(len(file_content))
        self._validate_pdf_signature(file_content)
        self._validate_pdf_structure(file_content)

        document_id = self._generate_document_id()
        storage_filename = self._generate_storage_filename(document_id, filename)

        stored_path = None
        try:
            stored_path = await self._store_file(storage_filename, file_content)
            metadata = await self._create_metadata(
                document_id=document_id,
                original_filename=filename,
                storage_filename=storage_filename,
                file_type=content_type or "application/pdf",
                file_size=len(file_content),
                uploaded_by=uploaded_by,
            )
            await self._insert_metadata(metadata)
            return metadata
        except Exception:
            if stored_path and stored_path.exists():
                try:
                    stored_path.unlink()
                except OSError:
                    pass
            raise

    async def list_documents(
        self,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[dict], int]:
        """
        List documents with pagination.

        Args:
            page: Page number (1-indexed)
            page_size: Documents per page

        Returns:
            Tuple of (documents list, total count)
        """
        page = max(1, page)
        page_size = max(1, min(page_size, 100))
        skip = (page - 1) * page_size

        try:
            total = await self.collection.count_documents({})
            cursor = self.collection.find({}).sort("upload_timestamp", -1).skip(skip).limit(page_size)
            documents = await cursor.to_list(length=page_size)
            return documents, total
        except PyMongoError as exc:
            logger.error("Failed to list documents: %s", exc)
            raise DocumentServiceError("Failed to retrieve documents") from exc

    async def delete_document(self, document_id: str) -> bool:
        """
        Delete document metadata and associated file.

        Args:
            document_id: Document identifier

        Returns:
            True if document was deleted

        Raises:
            DocumentNotFoundError: If document doesn't exist
            DocumentServiceError: If deletion fails
        """
        metadata = await self._get_metadata(document_id)
        if not metadata:
            raise DocumentNotFoundError(f"Document {document_id} not found")

        storage_filename = metadata.get("storage_filename")
        file_deleted = False
        if storage_filename:
            try:
                file_deleted = await self._delete_file(storage_filename)
            except StorageError:
                logger.error("Failed to delete file for document %s, proceeding with metadata deletion", document_id)

        metadata_deleted = await self._delete_metadata(document_id)

        if not metadata_deleted:
            raise DocumentServiceError("Failed to delete document metadata")

        return True

    async def create_indexes(self) -> None:
        """Create necessary MongoDB indexes."""
        try:
            await self.collection.create_index("document_id", unique=True)
            await self.collection.create_index("uploaded_by")
            await self.collection.create_index("upload_timestamp")
            logger.info("Document collection indexes created")
        except PyMongoError as exc:
            logger.error("Failed to create document indexes: %s", exc)
            raise DocumentServiceError("Failed to create indexes") from exc