import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import settings

async def reset_mongodb():
    client = AsyncIOMotorClient(settings.MONGODB_URL)
    db = client[settings.DATABASE_NAME]

    # Keep only the 5 deduplicated documents
    keep_doc_ids = {
        'doc_07166fa52b9841cf8ce385f7',
        'doc_4e3c3527c7244ebeb3fc692e',
        'doc_23b7363952504c548973d0f3',
        'doc_1e66fd28824c4610abf0e2ca',
        'doc_7bf37c5268d14677ac71f413',
    }
    keep_chunk_ids = {
        'doc_07166fa52b9841cf8ce385f7_chunk_0',
        'doc_4e3c3527c7244ebeb3fc692e_chunk_0',
        'doc_23b7363952504c548973d0f3_chunk_0',
        'doc_1e66fd28824c4610abf0e2ca_chunk_0',
        'doc_7bf37c5268d14677ac71f413_chunk_0',
    }

    # Delete documents not in keep list
    result = await db.documents.delete_many({'document_id': {'$nin': list(keep_doc_ids)}})
    print('Deleted', result.deleted_count, 'documents')

    # Delete chunks not in keep list
    result = await db.document_chunks.delete_many({'chunk_id': {'$nin': list(keep_chunk_ids)}})
    print('Deleted', result.deleted_count, 'chunks')

    # Verify
    docs = await db.documents.find({}).to_list(None)
    chunks = await db.document_chunks.find({}).to_list(None)
    print('Remaining documents:', len(docs))
    for d in docs:
        print(' ', d['document_id'], '|', d['original_filename'])
    print('Remaining chunks:', len(chunks))
    for c in chunks:
        print(' ', c['chunk_id'], '|', c['document_id'])

    client.close()

asyncio.run(reset_mongodb())