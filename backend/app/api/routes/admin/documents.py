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
from app.services.rag_ingestion import RAGIngestionService, RAGIngestionError

logger = logging.getLogger("meditrust.api.admin.documents")

router = APIRouter(prefix="/admin/documents", tags=["Admin Documents"])


async def get_document_service(
    db: Annotated[AsyncDatabase, Depends(get_db)],
) -> DocumentService:
    """Get DocumentService instance with database connection."""
    return DocumentService(db)


async def get_rag_ingestion_service(
    db: Annotated[AsyncDatabase, Depends(get_db)],
) -> RAGIngestionService:
    """Get RAGIngestionService instance with database connection."""
    service = RAGIngestionService(db)
    await service.create_indexes()
    return service


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


@router.post(
    "/{document_id}/ingest",
    response_model=dict,
    summary="Trigger RAG ingestion for a document",
    responses={
        200: {"description": "Ingestion completed"},
        202: {"description": "Ingestion started"},
        401: {"model": DocumentErrorResponse, "description": "Not authenticated"},
        403: {"model": DocumentErrorResponse, "description": "Insufficient permissions"},
        404: {"model": DocumentErrorResponse, "description": "Document not found"},
        500: {"model": DocumentErrorResponse, "description": "Ingestion failed"},
    },
)
async def ingest_document(
    document_id: str,
    ingestion_service: Annotated[RAGIngestionService, Depends(get_rag_ingestion_service)],
    current_user: Annotated[UserResponse, Depends(get_current_active_user)],
    _admin: Annotated[None, Depends(require_admin)],
) -> dict:
    """
    Trigger RAG ingestion for an uploaded document.

    - Only admin users can trigger ingestion
    - Processes document: loads, splits, embeds, stores in vector database
    - Returns ingestion result with chunk count and status
    """
    try:
        result = await ingestion_service.ingest_document(document_id)
        if result.status == "completed":
            return {
                "message": "Document ingested successfully",
                "document_id": result.document_id,
                "chunks_created": result.chunks_created,
                "status": result.status,
            }
        elif result.status == "failed":
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.error or "Ingestion failed"
            )
        else:
            return {
                "message": "Ingestion completed",
                "document_id": result.document_id,
                "chunks_created": result.chunks_created,
                "status": result.status,
            }
    except RAGIngestionError as exc:
        logger.error("Ingestion error for document %s: %s", document_id, exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))
    except DocumentNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    except Exception as exc:
        logger.exception("Unexpected error during ingestion: %s", exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@router.get(
    "/{document_id}/ingestion-status",
    response_model=dict,
    summary="Get RAG ingestion status for a document",
    responses={
        200: {"description": "Ingestion status retrieved"},
        401: {"model": DocumentErrorResponse, "description": "Not authenticated"},
        403: {"model": DocumentErrorResponse, "description": "Insufficient permissions"},
        404: {"model": DocumentErrorResponse, "description": "Document not found"},
    },
)
async def get_ingestion_status(
    document_id: str,
    ingestion_service: Annotated[RAGIngestionService, Depends(get_rag_ingestion_service)],
    current_user: Annotated[UserResponse, Depends(get_current_active_user)],
    _admin: Annotated[None, Depends(require_admin)],
) -> dict:
    """
    Get RAG ingestion status for a document.

    - Only admin users can check ingestion status
    - Returns processing status, chunk count, and any error
    """
    try:
        status_info = await ingestion_service.get_ingestion_status(document_id)
        if status_info is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
        return status_info
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error getting ingestion status: %s", exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve status")


@router.post(
    "/{document_id}/reingest",
    response_model=dict,
    summary="Re-ingest a document (replace existing chunks)",
    responses={
        200: {"description": "Re-ingestion completed"},
        401: {"model": DocumentErrorResponse, "description": "Not authenticated"},
        403: {"model": DocumentErrorResponse, "description": "Insufficient permissions"},
        404: {"model": DocumentErrorResponse, "description": "Document not found"},
        500: {"model": DocumentErrorResponse, "description": "Re-ingestion failed"},
    },
)
async def reingest_document(
    document_id: str,
    ingestion_service: Annotated[RAGIngestionService, Depends(get_rag_ingestion_service)],
    current_user: Annotated[UserResponse, Depends(get_current_active_user)],
    _admin: Annotated[None, Depends(require_admin)],
) -> dict:
    """
    Re-ingest a document (delete old chunks, create new ones).

    - Only admin users can trigger re-ingestion
    - Useful after changing embedding model or chunk settings
    """
    try:
        result = await ingestion_service.reingest_document(document_id)
        return {
            "message": "Document re-ingested successfully",
            "document_id": result.document_id,
            "chunks_created": result.chunks_created,
            "status": result.status,
        }
    except RAGIngestionError as exc:
        logger.error("Re-ingestion error for document %s: %s", document_id, exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))
    except DocumentNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    except Exception as exc:
        logger.exception("Unexpected error during re-ingestion: %s", exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


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
    ingestion_service: Annotated[RAGIngestionService, Depends(get_rag_ingestion_service)],
    current_user: Annotated[UserResponse, Depends(get_current_active_user)],
    _admin: Annotated[None, Depends(require_admin)],
) -> DocumentDeleteResponse:
    """
    Delete a document and its associated file.

    - Only admin users can delete documents
    - Validates document ID format
    - Removes both the stored PDF file and MongoDB metadata
    - Also removes RAG chunks from vector store
    - Handles missing files gracefully
    - Prevents path traversal attacks

    Returns:
        Deletion confirmation with document ID
    """
    try:
        # Delete RAG chunks first
        await ingestion_service.delete_document_chunks(document_id)

        # Delete document metadata and file
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