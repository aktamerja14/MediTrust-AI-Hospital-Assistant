from app.database.connection import db_manager
import asyncio

async def check():
    await db_manager.connect()
    db = db_manager.db
    docs = await db['doctors'].find().to_list(length=10)
    with open('db_photo_paths.txt', 'w', encoding='utf-8') as f:
        for doc in docs:
            f.write(f"doctor_id: {doc['doctor_id']}\n")
            f.write(f"photo_path: {doc['photo_path']}\n")
            f.write('---\n')
    await db_manager.disconnect()

asyncio.run(check())