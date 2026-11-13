"""RAG Document Ingestion Service.

Processes uploaded documents: loads, splits, embeds, and stores in vector database.
Tracks processing status in MongoDB.
"""
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

from pymongo.asynchronous.database import AsyncDatabase
from pymongo.errors import PyMongoError

from app.ai.embeddings.embedding_factory import EmbeddingFactory
from app.ai.rag.document_loader import DocumentLoader, DocumentLoaderError
from app.ai.rag.text_splitter import TextSplitter, TextChunk
from app.ai.vectorstore.base import VectorDocument
from app.ai.vectorstore.factory import VectorStoreFactory
from app.core.config import settings


def _utc_now() -> datetime:
    """Return current UTC time as timezone-aware datetime."""
    return datetime.now(timezone.utc)
from app.services.document_service import DocumentService

logger = logging.getLogger("meditrust.services.rag_ingestion")

# Collection for document chunk metadata
DOCUMENT_CHUNKS_COLLECTION = "document_chunks"

# Processing statuses
STATUS_PENDING = "pending"
STATUS_PROCESSING = "processing"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"


@dataclass
class IngestionResult:
    """Result of document ingestion."""
    document_id: str
    chunks_created: int
    status: str
    error: Optional[str] = None


class RAGIngestionError(Exception):
    """Exception for ingestion errors."""
    pass


class RAGIngestionService:
    """Service for ingesting documents into the RAG pipeline."""

    def __init__(self, db: AsyncDatabase):
        self.db = db
        self.chunks_collection = db[DOCUMENT_CHUNKS_COLLECTION]
        self.document_service = DocumentService(db)
        self.document_loader = DocumentLoader()
        self.text_splitter = TextSplitter()
        self._embedding_provider = None
        self._vector_store = None

    @property
    def embedding_provider(self):
        if self._embedding_provider is None:
            self._embedding_provider = EmbeddingFactory.get_instance()
        return self._embedding_provider

    @property
    def vector_store(self):
        if self._vector_store is None:
            self._vector_store = VectorStoreFactory.get_instance()
        return self._vector_store

    async def create_indexes(self) -> None:
        """Create necessary MongoDB indexes."""
        try:
            await self.chunks_collection.create_index("document_id")
            await self.chunks_collection.create_index("chunk_id", unique=True)
            await self.chunks_collection.create_index("processing_status")
            logger.info("RAG ingestion indexes created")
        except PyMongoError as e:
            logger.error("Failed to create RAG indexes: %s", e)
            raise RAGIngestionError("Failed to create indexes") from e

    async def ingest_document(self, document_id: str) -> IngestionResult:
        """Ingest a document into the RAG pipeline.

        Args:
            document_id: Document ID from document service

        Returns:
            IngestionResult with status and chunk count
        """
        # Get document metadata
        doc_meta = await self.document_service._get_metadata(document_id)
        if not doc_meta:
            raise RAGIngestionError(f"Document {document_id} not found")

        # Check if already processed
        existing_chunks = await self.chunks_collection.count_documents(
            {"document_id": document_id}
        )
        if existing_chunks > 0:
            logger.info("Document %s already ingested (%d chunks)", document_id, existing_chunks)
            return IngestionResult(
                document_id=document_id,
                chunks_created=existing_chunks,
                status=STATUS_COMPLETED,
            )

        # Update status to processing
        await self._update_processing_status(document_id, STATUS_PROCESSING)

        try:
            # Load document
            pages = self.document_loader.load_document(doc_meta["storage_filename"])

            if not pages:
                raise RAGIngestionError("Document has no extractable text")

            # Split into chunks
            all_chunks: List[TextChunk] = []
            for page in pages:
                base_metadata = {
                    "document_id": document_id,
                    "source": doc_meta["original_filename"],
                    "source_filename": doc_meta["original_filename"],
                    "storage_filename": doc_meta["storage_filename"],
                    "page_number": page["page_number"],
                    "file_type": doc_meta["file_type"],
                }
                chunks = self.text_splitter.split_text(page["content"], base_metadata)
                all_chunks.extend(chunks)

            if not all_chunks:
                raise RAGIngestionError("No chunks created from document")

            logger.info("Split document %s into %d chunks", document_id, len(all_chunks))

            # Generate embeddings
            chunk_texts = [chunk.content for chunk in all_chunks]
            embeddings = self.embedding_provider.embed_texts(chunk_texts)

            # Create vector documents
            vector_docs = []
            for i, (chunk, embedding) in enumerate(zip(all_chunks, embeddings)):
                chunk_id = f"{document_id}_chunk_{i}"
                vector_doc = VectorDocument(
                    id=chunk_id,
                    content=chunk.content,
                    embedding=embedding,
                    metadata=chunk.metadata,
                )
                vector_docs.append(vector_doc)

                # Store chunk metadata in MongoDB
                chunk_meta = {
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "content": chunk.content,
                    "chunk_index": i,
                    "page_number": chunk.metadata.get("page_number"),
                    "metadata": chunk.metadata,
                    "embedding_model": settings.EMBEDDING_MODEL,
                    "processing_status": STATUS_COMPLETED,
                    "created_at": _utc_now(),
                }
                await self.chunks_collection.insert_one(chunk_meta)

            # Add to vector store
            self.vector_store.add_documents(vector_docs)
            self.vector_store.persist()

            # Update document processing status
            await self._update_processing_status(document_id, STATUS_COMPLETED)

            logger.info("Successfully ingested document %s: %d chunks", document_id, len(vector_docs))
            return IngestionResult(
                document_id=document_id,
                chunks_created=len(vector_docs),
                status=STATUS_COMPLETED,
            )

        except (DocumentLoaderError, RAGIngestionError) as e:
            await self._update_processing_status(document_id, STATUS_FAILED, str(e))
            logger.error("Ingestion failed for %s: %s", document_id, e)
            return IngestionResult(
                document_id=document_id,
                chunks_created=0,
                status=STATUS_FAILED,
                error=str(e),
            )
        except Exception as e:
            await self._update_processing_status(document_id, STATUS_FAILED, str(e))
            logger.exception("Unexpected ingestion error for %s: %s", document_id, e)
            return IngestionResult(
                document_id=document_id,
                chunks_created=0,
                status=STATUS_FAILED,
                error=f"Unexpected error: {str(e)}",
            )

    async def _update_processing_status(
        self,
        document_id: str,
        status: str,
        error: Optional[str] = None,
    ) -> None:
        """Update document processing status in MongoDB."""
        try:
            update = {
                "processing_status": status,
                "processing_updated_at": _utc_now(),
            }
            if error:
                update["processing_error"] = error

            await self.document_service.collection.update_one(
                {"document_id": document_id},
                {"$set": update},
            )
        except PyMongoError as e:
            logger.error("Failed to update processing status: %s", e)

    async def delete_document_chunks(self, document_id: str) -> int:
        """Delete all chunks for a document from vector store and MongoDB."""
        # Get chunk IDs
        chunk_ids = []
        async for chunk in self.chunks_collection.find({"document_id": document_id}):
            chunk_ids.append(chunk["chunk_id"])

        # Delete from vector store
        for chunk_id in chunk_ids:
            self.vector_store.delete_document(chunk_id)
        self.vector_store.persist()

        # Delete from MongoDB
        result = await self.chunks_collection.delete_many({"document_id": document_id})
        logger.info("Deleted %d chunks for document %s", result.deleted_count, document_id)
        return result.deleted_count

    async def get_ingestion_status(self, document_id: str) -> Optional[dict]:
        """Get ingestion status for a document."""
        doc_meta = await self.document_service._get_metadata(document_id)
        if not doc_meta:
            return None

        chunks_count = await self.chunks_collection.count_documents(
            {"document_id": document_id}
        )

        return {
            "document_id": document_id,
            "processing_status": doc_meta.get("processing_status", STATUS_PENDING),
            "processing_error": doc_meta.get("processing_error"),
            "chunks_count": chunks_count,
            "updated_at": doc_meta.get("processing_updated_at"),
        }

    async def reingest_document(self, document_id: str) -> IngestionResult:
        """Re-ingest a document (delete old chunks, create new ones)."""
        await self.delete_document_chunks(document_id)
        # Reset processing status
        await self._update_processing_status(document_id, STATUS_PENDING)
        return await self.ingest_document(document_id)