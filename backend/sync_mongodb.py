import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import settings
import json

async def sync_mongodb():
    client = AsyncIOMotorClient(settings.MONGODB_URL)
    db = client[settings.DATABASE_NAME]

    with open('data/vector_store/metadata.json', 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    docs_by_id = {}
    for chunk_id, meta in metadata.items():
        doc_id = meta['metadata']['document_id']
        if doc_id not in docs_by_id:
            docs_by_id[doc_id] = {
                'document_id': doc_id,
                'original_filename': meta['metadata']['source_filename'],
                'storage_filename': meta['metadata']['storage_filename'],
                'file_type': meta['metadata']['file_type'],
                'processing_status': 'completed',
                'chunks_count': 0
            }
        docs_by_id[doc_id]['chunks_count'] += 1

    print('=== SYNCING MONGODB DOCUMENTS ===')
    for doc_id, doc_meta in docs_by_id.items():
        existing = await db.documents.find_one({'document_id': doc_id})
        if existing:
            print('Document {} exists, updating chunks_count={}'.format(doc_id, doc_meta['chunks_count']))
            await db.documents.update_one(
                {'document_id': doc_id},
                {'$set': {'chunks_count': doc_meta['chunks_count'], 'processing_status': 'completed'}}
            )
        else:
            print('Inserting new document: {} ({})'.format(doc_id, doc_meta['original_filename']))
            await db.documents.insert_one(doc_meta)

    print('\n=== SYNCING MONGODB DOCUMENT_CHUNKS ===')
    for chunk_id, meta in metadata.items():
        chunk_doc = {
            'chunk_id': chunk_id,
            'document_id': meta['metadata']['document_id'],
            'content': meta['content'],
            'chunk_index': meta['metadata']['chunk_index'],
            'page_number': meta['metadata']['page_number'],
            'metadata': meta['metadata'],
            'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
            'processing_status': 'completed'
        }
        existing = await db.document_chunks.find_one({'chunk_id': chunk_id})
        if existing:
            print('Chunk {} exists, updating'.format(chunk_id))
            await db.document_chunks.update_one({'chunk_id': chunk_id}, {'$set': chunk_doc})
        else:
            print('Inserting new chunk: {}'.format(chunk_id))
            await db.document_chunks.insert_one(chunk_doc)

    docs = await db.documents.find({}).to_list(None)
    chunks = await db.document_chunks.find({}).to_list(None)
    print('\n=== FINAL MONGODB STATE ===')
    print('Documents: {}'.format(len(docs)))
    for d in docs:
        print('  {} | {} | chunks={} | status={}'.format(
            d['document_id'], d['original_filename'], d.get('chunks_count', 0), d.get('processing_status')))
    print('Chunks: {}'.format(len(chunks)))
    for c in chunks:
        print('  {} | {}'.format(c['chunk_id'], c['document_id']))

    client.close()

asyncio.run(sync_mongodb())