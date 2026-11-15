import asyncio
from pymongo import AsyncMongoClient
from app.core.config import settings

async def main():
    client = AsyncMongoClient(settings.MONGODB_URL, serverSelectionTimeoutMS=5000)
    db = client[settings.DATABASE_NAME]
    
    docs = await db.documents.find({}).to_list(length=100)
    print(f'Total documents in MongoDB: {len(docs)}')
    for doc in docs:
        print(f'  Document ID: {doc["document_id"]}, Filename: {doc["original_filename"]}, Status: {doc.get("processing_status", "N/A")}')
    
    chunks = await db.document_chunks.find({}).to_list(length=500)
    print(f'\nTotal chunks in MongoDB: {len(chunks)}')
    
    from collections import Counter
    chunk_counts = Counter(c['document_id'] for c in chunks)
    for doc_id, count in chunk_counts.items():
        print(f'  {doc_id}: {count} chunks')
    
    chunk_ids = [c['chunk_id'] for c in chunks]
    dup_counts = Counter(chunk_ids)
    duplicates = {k: v for k, v in dup_counts.items() if v > 1}
    if duplicates:
        print(f'\nDUPLICATE chunk_ids found: {duplicates}')
    else:
        print('\nNo duplicate chunk_ids found')
    
    await client.close()

asyncio.run(main())