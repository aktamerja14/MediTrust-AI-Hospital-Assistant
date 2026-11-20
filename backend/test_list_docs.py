import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import settings
from app.services.document_service import DocumentService

async def test_list():
    client = AsyncIOMotorClient(settings.MONGODB_URL)
    db = client[settings.DATABASE_NAME]

    service = DocumentService(db)
    docs, total = await service.list_documents(page=1, page_size=20)
    print(f'Total: {total}, Retrieved: {len(docs)}')
    for doc in docs:
        print('  {}: file_size={}, upload_timestamp={}, uploaded_by={}'.format(
            doc['document_id'],
            doc.get('file_size'),
            doc.get('upload_timestamp'),
            doc.get('uploaded_by')
        ))

    client.close()

asyncio.run(test_list())