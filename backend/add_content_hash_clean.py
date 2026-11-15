import asyncio
import hashlib
from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import settings

async def add_content_hashes():
    client = AsyncIOMotorClient(settings.MONGODB_URL)
    db = client[settings.DATABASE_NAME]

    chunks = await db.document_chunks.find({}).to_list(None)
    print('Found', len(chunks), 'chunks')

    for chunk in chunks:
        content = chunk['content']
        content_hash = hashlib.sha256(content.strip().encode('utf-8')).hexdigest()
        await db.document_chunks.update_one(
            {'chunk_id': chunk['chunk_id']},
            {'$set': {'content_hash': content_hash}}
        )
        print('  Updated:', chunk['chunk_id'], 'hash=', content_hash[:16])

    client.close()

asyncio.run(add_content_hashes())