"""Document management request/response schemas."""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class DocumentMetadata(BaseModel):
    """Document metadata stored in MongoDB."""
    document_id: str = Field(..., description="Unique document identifier")
    original_filename: str = Field(..., description="Original uploaded filename")
    storage_filename: str = Field(..., description="Server-side storage filename")
    file_type: str = Field(..., description="File MIME type")
    file_size: int = Field(..., description="File size in bytes")
    upload_timestamp: datetime = Field(..., description="Upload timestamp")
    processing_status: str = Field(default="uploaded", description="Processing status")
    uploaded_by: str = Field(..., description="User ID who uploaded the document")

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "document_id": "doc_507f1f77bcf86cd799439011",
                "original_filename": "hospital_policy.pdf",
                "storage_filename": "doc_507f1f77bcf86cd799439011.pdf",
                "file_type": "application/pdf",
                "file_size": 1024000,
                "upload_timestamp": "2024-01-15T10:30:00Z",
                "processing_status": "uploaded",
                "uploaded_by": "507f1f77bcf86cd799439011"
            }
        }
    )


class DocumentUploadResponse(BaseModel):
    """Response for successful document upload."""
    document_id: str = Field(..., description="Unique document identifier")
    original_filename: str = Field(..., description="Original uploaded filename")
    file_size: int = Field(..., description="File size in bytes")
    upload_timestamp: datetime = Field(..., description="Upload timestamp")
    processing_status: str = Field(..., description="Processing status")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "document_id": "doc_507f1f77bcf86cd799439011",
                "original_filename": "hospital_policy.pdf",
                "file_size": 1024000,
                "upload_timestamp": "2024-01-15T10:30:00Z",
                "processing_status": "uploaded"
            }
        }
    )


class DocumentListResponse(BaseModel):
    """Paginated document list response."""
    documents: list[DocumentMetadata]
    total: int = Field(..., description="Total number of documents")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Number of documents per page")
    total_pages: int = Field(..., description="Total number of pages")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "documents": [
                    {
                        "document_id": "doc_507f1f77bcf86cd799439011",
                        "original_filename": "hospital_policy.pdf",
                        "storage_filename": "doc_507f1f77bcf86cd799439011.pdf",
                        "file_type": "application/pdf",
                        "file_size": 1024000,
                        "upload_timestamp": "2024-01-15T10:30:00Z",
                        "processing_status": "uploaded",
                        "uploaded_by": "507f1f77bcf86cd799439011"
                    }
                ],
                "total": 1,
                "page": 1,
                "page_size": 20,
                "total_pages": 1
            }
        }
    )


class DocumentDeleteResponse(BaseModel):
    """Response for successful document deletion."""
    message: str = Field(..., description="Deletion confirmation message")
    document_id: str = Field(..., description="Deleted document identifier")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "message": "Document deleted successfully",
                "document_id": "doc_507f1f77bcf86cd799439011"
            }
        }
    )


class DocumentErrorResponse(BaseModel):
    """Error response for document operations."""
    detail: str

    model_config = ConfigDict(
        json_schema_extra={"example": {"detail": "Document not found"}}
    )