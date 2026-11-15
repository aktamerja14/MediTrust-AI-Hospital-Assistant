import asyncio
import uuid
import json
import os
import faiss
import pickle
from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import settings
from app.services.rag_ingestion import RAGIngestionService
from app.services.document_service import DocumentService
from app.ai.vectorstore.factory import VectorStoreFactory
from app.ai.embeddings.embedding_factory import EmbeddingFactory

async def run_all_tests():
    client = AsyncIOMotorClient(settings.MONGODB_URL)
    db = client[settings.DATABASE_NAME]

    service = RAGIngestionService(db)
    await service.create_indexes()
    doc_service = DocumentService(db)

    print('=== INITIAL STATE ===')
    VectorStoreFactory.reset()
    store = VectorStoreFactory.get_instance()
    stats = store.get_stats()
    print('Vector store:', stats)

    # TEST 1: Backend restart
    print('\n=== TEST 1: Backend Restart ===')
    VectorStoreFactory.reset()
    store = VectorStoreFactory.get_instance()
    stats = store.get_stats()
    print('Vector store:', stats)
    assert stats['total_vectors'] == 5, 'Expected 5, got ' + str(stats['total_vectors'])
    assert stats['tracked_documents'] == 5, 'Expected 5, got ' + str(stats['tracked_documents'])
    print('TEST 1: PASS')

    # TEST 2: Ingest same JSON twice
    print('\n=== TEST 2: Ingest same JSON content twice ===')
    new_doc_id = 'doc_new_qa_test_' + uuid.uuid4().hex[:12]
    await doc_service.collection.insert_one({
        'document_id': new_doc_id,
        'original_filename': 'qa.json',
        'storage_filename': 'doc_07166fa52b9841cf8ce385f7.json',
        'file_type': 'application/json',
        'processing_status': 'pending',
        'chunks_count': 0,
    })
    result = await service.ingest_document(new_doc_id)
    print('Result:', result.status, 'chunks_created:', result.chunks_created)
    VectorStoreFactory.reset()
    store = VectorStoreFactory.get_instance()
    stats = store.get_stats()
    print('Vector store:', stats)
    assert stats['total_vectors'] == 5, 'Expected 5, got ' + str(stats['total_vectors'])
    print('TEST 2: PASS')

    # TEST 3: Ingest different JSON
    print('\n=== TEST 3: Ingest different JSON content ===')
    new_json_content = [
        {'question': 'What is the hospital name?', 'answer': 'MediTrust Hospital'},
        {'question': 'What are the departments?', 'answer': 'Cardiology, Neurology'}
    ]
    new_storage_filename = 'doc_new_faq_' + uuid.uuid4().hex[:8] + '.json'
    storage_path = os.path.join('data/documents', new_storage_filename)
    with open(storage_path, 'w', encoding='utf-8') as f:
        json.dump(new_json_content, f, ensure_ascii=False, indent=2)

    new_doc_id2 = 'doc_new_faq_' + uuid.uuid4().hex[:12]
    await doc_service.collection.insert_one({
        'document_id': new_doc_id2,
        'original_filename': 'new_faq.json',
        'storage_filename': new_storage_filename,
        'file_type': 'application/json',
        'processing_status': 'pending',
        'chunks_count': 0,
    })
    result = await service.ingest_document(new_doc_id2)
    print('Result:', result.status, 'chunks_created:', result.chunks_created)
    VectorStoreFactory.reset()
    store = VectorStoreFactory.get_instance()
    stats = store.get_stats()
    print('Vector store:', stats)
    assert stats['total_vectors'] == 6, 'Expected 6, got ' + str(stats['total_vectors'])
    print('TEST 3: PASS')

    # TEST 4: Re-ingest updated document
    print('\n=== TEST 4: Re-ingest updated document ===')
    updated_content = [
        {'question': 'What is the hospital name?', 'answer': 'MediTrust Hospital'},
        {'question': 'What are the departments?', 'answer': 'Cardiology, Neurology, Oncology'},
        {'question': 'What are visiting hours?', 'answer': '9 AM to 6 PM'}
    ]
    with open(storage_path, 'w', encoding='utf-8') as f:
        json.dump(updated_content, f, ensure_ascii=False, indent=2)

    await doc_service.collection.update_one(
        {'document_id': new_doc_id2},
        {'$set': {'storage_filename': new_storage_filename, 'processing_status': 'pending'}}
    )

    result = await service.reingest_document(new_doc_id2)
    print('Result:', result.status, 'chunks_created:', result.chunks_created)
    VectorStoreFactory.reset()
    store = VectorStoreFactory.get_instance()
    stats = store.get_stats()
    print('Vector store:', stats)
    assert stats['total_vectors'] == 6, 'Expected 6, got ' + str(stats['total_vectors'])
    print('TEST 4: PASS')

    # TEST 5: Backend restart after all operations
    print('\n=== TEST 5: Backend Restart ===')
    VectorStoreFactory.reset()
    store = VectorStoreFactory.get_instance()
    stats = store.get_stats()
    print('Vector store:', stats)
    assert stats['total_vectors'] == 6, 'Expected 6, got ' + str(stats['total_vectors'])
    assert stats['tracked_documents'] == 6, 'Expected 6, got ' + str(stats['tracked_documents'])
    print('TEST 5: PASS')

    # TEST 6: Retrieval
    print('\n=== TEST 6: Retrieval ===')
    embedder = EmbeddingFactory.get_instance()

    eng_query = 'What are visiting hours?'
    eng_embedding = embedder.embed_text(eng_query)
    results = store.search(eng_embedding, top_k=3)
    print('English query:', eng_query)
    assert len(results) > 0
    assert results[0].document.metadata.get('source_filename') == 'qa.json'
    print('  Top result:', results[0].document.metadata.get('source_filename'), '- PASS')

    hosp_query = 'MediTrust hospital departments'
    hosp_embedding = embedder.embed_text(hosp_query)
    results = store.search(hosp_embedding, top_k=3)
    print('Hospital query:', hosp_query)
    assert len(results) > 0
    assert results[0].document.metadata.get('source_filename') == 'hospital.json'
    print('  Top result:', results[0].document.metadata.get('source_filename'), '- PASS')

    print('TEST 6: PASS')

    # TEST 7: Count verification
    print('\n=== TEST 7: Count Verification ===')
    index = faiss.read_index('data/vector_store/index.faiss')
    with open('data/vector_store/metadata.json', 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    with open('data/vector_store/id_map.pkl', 'rb') as f:
        id_map = pickle.load(f)

    print('FAISS vectors:', index.ntotal)
    print('Metadata chunks:', len(metadata))
    print('ID map entries:', len(id_map))
    assert index.ntotal == len(metadata) == len(id_map) == 6
    print('TEST 7: PASS')

    print('\n=== ALL TESTS PASSED ===')
    client.close()

asyncio.run(run_all_tests())