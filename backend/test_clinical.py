import asyncio
import sys
sys.stdout.reconfigure(encoding="utf-8")
from app.database.connection import db_manager

async def test():
    await db_manager.connect()
    db = db_manager.db
    
    for source in ["Clinical Guidelines.txt", "doctor_dataset.txt"]:
        cursor = db["document_chunks"].find({
            "metadata.source": source
        }).limit(3)
        
        print("=== Source:", source, "===")
        async for chunk in cursor:
            print("Content:", chunk["content"][:500])
            print("---")

asyncio.run(test())