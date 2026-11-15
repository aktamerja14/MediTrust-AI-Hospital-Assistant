import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import settings

async def cleanup():
    client = AsyncIOMotorClient(settings.MONGODB_URL)
    db = client[settings.DATABASE_NAME]

    # Delete test documents
    test_docs = await db.documents.find({'document_id': {'$regex': 'doc_new_qa_test'}}).to_list(None)
    for doc in test_docs:
        print('Deleting document:', doc['document_id'])
        await db.documents.delete_one({'document_id': doc['document_id']})
        await db.document_chunks.delete_many({'document_id': doc['document_id']})

    # Delete test chunks
    await db.document_chunks.delete_many({'chunk_id': {'$regex': 'doc_new_qa_test'}})

    # Verify
    docs = await db.documents.find({}).to_list(None)
    chunks = await db.document_chunks.find({}).to_list(None)
    print('Remaining documents:', len(docs))
    for d in docs:
        print(' ', d['document_id'], '|', d['original_filename'])
    print('Remaining chunks:', len(chunks))
    for c in chunks:
        print(' ', c['chunk_id'], '|', c['document_id'])

    print('Cleanup done')
    client.close()

asyncio.run(cleanup())