import asyncio
import uuid
import json
import os
from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import settings
from app.services.rag_ingestion import RAGIngestionService
from app.services.document_service import DocumentService

async def test_reingest():
    client = AsyncIOMotorClient(settings.MONGODB_URL)
    db = client[settings.DATABASE_NAME]

    service = RAGIngestionService(db)
    await service.create_indexes()
    doc_service = DocumentService(db)

    print('=== TEST 4: Re-ingest updated document ===')

    doc_id = 'doc_new_faq_f50193704d62'

    updated_content = [
        {'question': 'What is the hospital name?', 'answer': 'MediTrust Hospital'},
        {'question': 'What are the departments?', 'answer': 'Cardiology, Neurology, Oncology'},
        {'question': 'What are visiting hours?', 'answer': '9 AM to 6 PM'}
    ]

    storage_filename = 'doc_new_faq_f50193704d62.json'
    storage_path = os.path.join('data/documents', storage_filename)
    with open(storage_path, 'w', encoding='utf-8') as f:
        json.dump(updated_content, f, ensure_ascii=False, indent=2)

    await doc_service.collection.update_one(
        {'document_id': doc_id},
        {'$set': {'storage_filename': storage_filename, 'processing_status': 'pending'}}
    )

    result = await service.reingest_document(doc_id)
    print('Re-ingest updated document result:', result.status, 'chunks_created:', result.chunks_created)

    from app.ai.vectorstore.factory import VectorStoreFactory
    VectorStoreFactory.reset()
    store = VectorStoreFactory.get_instance()
    stats = store.get_stats()
    print('Vector store stats:', stats)
    print('TEST 4:', 'PASS' if stats['total_vectors'] >= 6 else 'FAIL - vectors:', stats['total_vectors'])

    client.close()

asyncio.run(test_reingest())