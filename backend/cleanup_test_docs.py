import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import settings

async def cleanup():
    client = AsyncIOMotorClient(settings.MONGODB_URL)
    db = client[settings.DATABASE_NAME]

    test_docs = await db.documents.find({'document_id': {'$regex': 'doc_new_qa_test'}}).to_list(None)
    for doc in test_docs:
        print('Deleting document:', doc['document_id'])
        await db.documents.delete_one({'document_id': doc['document_id']})
        await db.document_chunks.delete_many({'document_id': doc['document_id']})

    await db.document_chunks.delete_many({'chunk_id': {'$regex': 'doc_new_qa_test'}})

    print('Cleanup done')
    client.close()

asyncio.run(cleanup())