import asyncio
from datetime import datetime, timezone
from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import settings
from app.services.document_service import DocumentService
from app.schemas.document import DocumentMetadata

def _utc_now():
    return datetime.now(timezone.utc)

async def test_api_list():
    client = AsyncIOMotorClient(settings.MONGODB_URL)
    db = client[settings.DATABASE_NAME]

    service = DocumentService(db)
    docs, total = await service.list_documents(page=1, page_size=20)
    total_pages = (total + 20 - 1) // 20 if total > 0 else 0

    document_metadata = []
    for doc in docs:
        try:
            meta = DocumentMetadata(
                document_id=doc["document_id"],
                original_filename=doc["original_filename"],
                storage_filename=doc["storage_filename"],
                file_type=doc["file_type"],
                file_size=doc.get("file_size", 0) or 0,
                upload_timestamp=doc.get("upload_timestamp") or _utc_now(),
                processing_status=doc.get("processing_status", "uploaded"),
                uploaded_by=doc.get("uploaded_by", ""),
            )
            document_metadata.append(meta)
            print(f'  OK: {meta.document_id} -> file_size={meta.file_size}')
        except Exception as e:
            print(f'  ERROR: {doc.get("document_id")} -> {e}')

    print(f'Total documents processed: {len(document_metadata)}')
    client.close()

asyncio.run(test_api_list())