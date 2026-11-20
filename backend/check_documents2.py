import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').resolve()))
from dotenv import load_dotenv
load_dotenv(Path('.') / '.env')
from app.database.connection import DatabaseManager

async def check():
    db_manager = DatabaseManager()
    await db_manager.connect()
    db = db_manager.db
    
    docs = await db['documents'].find().to_list(length=100)
    for doc in docs:
        print(f'ID: {doc.get("_id")}')
        print(f'  Source: {doc.get("source")}')
        print(f'  Filename: {doc.get("filename")}')
        print(f'  Chunk count: {doc.get("chunk_count")}')
        print(f'  Status: {doc.get("status")}')
        print(f'  Content hash: {doc.get("content_hash")}')
        print()
    
    await db_manager.disconnect()

import asyncio
asyncio.run(check())