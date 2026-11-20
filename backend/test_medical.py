import asyncio
import sys
sys.stdout.reconfigure(encoding="utf-8")
from app.database.connection import db_manager

async def test():
    await db_manager.connect()
    db = db_manager.db
    
    # Search in medical_medicine_dataset chunks
    cursor = db["document_chunks"].find({
        "metadata.source": "medical_medicine_dataset.txt",
        "content": {"$regex": "diabetes", "$options": "i"}
    }).limit(5)
    
    async for chunk in cursor:
        print("Source:", chunk.get("metadata", {}).get("source"))
        print("Content:", chunk["content"][:300])
        print("---")

asyncio.run(test())