from app.database.connection import db_manager
import asyncio

async def check():
    await db_manager.connect()
    db = db_manager.db
    # Check doctors collection
    cursor = db['doctors'].find({})
    doctors = await cursor.to_list(length=100)
    for doc in doctors:
        print(f"doctor_id: {doc.get('doctor_id')}, name: {doc.get('name')}, specialization: {doc.get('specialization')}, photo_path: {doc.get('photo_path')}")

asyncio.run(check())