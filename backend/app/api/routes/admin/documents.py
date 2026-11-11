"""Admin document management API routes."""
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from pymongo.asynchronous.database import AsyncDatabase

from app.database.connection import get_db
from app.api.dependencies import get_current_active_user, require_admin
from app.schemas.auth import UserResponse
from app.schemas.document import (
    DocumentDeleteResponse,
    DocumentErrorResponse,
    DocumentListResponse,
    DocumentMetadata,
    DocumentUploadResponse,
)
from app.services.document_service import (
    DocumentNotFoundError,
    DocumentService,
    FileSizeError,
    InvalidFileError,
    StorageError,
)

logger = logging.getLogger("meditrust.api.admin.documents")

router = APIRouter(prefix="/admin/documents", tags=["Admin Documents"])


async def get_document_service(
    db: Annotated[AsyncDatabase, Depends(get_db)],
) -> DocumentService:
    """Get DocumentService instance with database connection."""
    return DocumentService(db)


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a PDF or TXT document",
    responses={
        201: {"description": "Document uploaded successfully"},
        400: {"model": DocumentErrorResponse, "description": "Invalid file"},
        401: {"model": DocumentErrorResponse, "description": "Not authenticated"},
        403: {"model": DocumentErrorResponse, "description": "Insufficient permissions"},
        413: {"model": DocumentErrorResponse, "description": "File too large"},
    },
)
async def upload_document(
    file: Annotated[UploadFile, File(description="PDF or TXT file to upload")],
    document_service: Annotated[DocumentService, Depends(get_document_service)],
    current_user: Annotated[UserResponse, Depends(get_current_active_user)],
    _admin: Annotated[None, Depends(require_admin)],
) -> DocumentUploadResponse:
    """
    Upload a PDF or plain text document to the hospital knowledge base.

    - Only admin users can upload documents
    - File must be a valid PDF (extension, MIME type, signature, structure) or plain text file
    - Maximum file size: 10 MB (configurable)
    - Empty files are rejected
    - Files are stored with secure server-generated names

    Returns:
        Document metadata including document_id, original filename, size, and timestamp
    """
    content = await file.read()
    filename = file.filename or ""
    content_type = file.content_type

    try:
        metadata = await document_service.upload_document(
            file_content=content,
            filename=filename,
            content_type=content_type,
            uploaded_by=current_user.id,
        )
        logger.info("Document uploaded by admin %s: %s", current_user.id, metadata["document_id"])
        return DocumentUploadResponse(
            document_id=metadata["document_id"],
            original_filename=metadata["original_filename"],
            file_size=metadata["file_size"],
            upload_timestamp=metadata["upload_timestamp"],
            processing_status=metadata["processing_status"],
        )
    except InvalidFileError as exc:
        logger.warning("Invalid file upload attempt by admin %s: %s", current_user.id, exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except FileSizeError as exc:
        logger.warning("File size exceeded by admin %s: %s", current_user.id, exc)
        raise HTTPException(status_code=status.HTTP_413_CONTENT_TOO_LARGE, detail=str(exc))
    except StorageError as exc:
        logger.error("Storage error during upload by admin %s: %s", current_user.id, exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to store document")
    except Exception as exc:
        logger.exception("Unexpected error during document upload: %s", exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List uploaded documents",
    responses={
        200: {"description": "Documents retrieved successfully"},
        401: {"model": DocumentErrorResponse, "description": "Not authenticated"},
        403: {"model": DocumentErrorResponse, "description": "Insufficient permissions"},
    },
)
async def list_documents(
    document_service: Annotated[DocumentService, Depends(get_document_service)],
    current_user: Annotated[UserResponse, Depends(get_current_active_user)],
    _admin: Annotated[None, Depends(require_admin)],
    page: Annotated[int, Query(ge=1, description="Page number (1-indexed)")] = 1,
    page_size: Annotated[int, Query(ge=1, le=100, description="Documents per page")] = 20,
) -> DocumentListResponse:
    """
    List all uploaded documents with pagination.

    - Only admin users can list documents
    - Supports pagination with page and page_size parameters
    - Returns document metadata (no file contents or internal paths)
    - Sorted by upload timestamp (newest first)

    Returns:
        Paginated list of document metadata
    """
    try:
        documents, total = await document_service.list_documents(page=page, page_size=page_size)
        total_pages = (total + page_size - 1) // page_size if total > 0 else 0

        document_metadata = [
            DocumentMetadata(
                document_id=doc["document_id"],
                original_filename=doc["original_filename"],
                storage_filename=doc["storage_filename"],
                file_type=doc["file_type"],
                file_size=doc["file_size"],
                upload_timestamp=doc["upload_timestamp"],
                processing_status=doc["processing_status"],
                uploaded_by=doc["uploaded_by"],
            )
            for doc in documents
        ]

        return DocumentListResponse(
            documents=document_metadata,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
        )
    except Exception as exc:
        logger.exception("Error listing documents: %s", exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve documents")


@router.delete(
    "/{document_id}",
    response_model=DocumentDeleteResponse,
    summary="Delete a document",
    responses={
        200: {"description": "Document deleted successfully"},
        401: {"model": DocumentErrorResponse, "description": "Not authenticated"},
        403: {"model": DocumentErrorResponse, "description": "Insufficient permissions"},
        404: {"model": DocumentErrorResponse, "description": "Document not found"},
    },
)
async def delete_document(
    document_id: str,
    document_service: Annotated[DocumentService, Depends(get_document_service)],
    current_user: Annotated[UserResponse, Depends(get_current_active_user)],
    _admin: Annotated[None, Depends(require_admin)],
) -> DocumentDeleteResponse:
    """
    Delete a document and its associated file.

    - Only admin users can delete documents
    - Validates document ID format
    - Removes both the stored PDF file and MongoDB metadata
    - Handles missing files gracefully
    - Prevents path traversal attacks

    Returns:
        Deletion confirmation with document ID
    """
    try:
        await document_service.delete_document(document_id)
        logger.info("Document deleted by admin %s: %s", current_user.id, document_id)
        return DocumentDeleteResponse(
            message="Document deleted successfully",
            document_id=document_id,
        )
    except DocumentNotFoundError as exc:
        logger.warning("Document not found for deletion by admin %s: %s", current_user.id, document_id)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except StorageError as exc:
        logger.error("Storage error during deletion by admin %s: %s", current_user.id, exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to delete document file")
    except Exception as exc:
        logger.exception("Unexpected error during document deletion: %s", exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")