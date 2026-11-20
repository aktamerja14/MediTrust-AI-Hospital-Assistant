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
        print(f'Source: {doc.get("source")}, Chunks: {doc.get("chunk_count")}, Status: {doc.get("status")}')
    
    await db_manager.disconnect()

import asyncio
asyncio.run(check())