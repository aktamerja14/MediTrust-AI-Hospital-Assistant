from app.database.connection import db_manager
import asyncio

async def check():
    await db_manager.connect()
    db = db_manager.db
    docs = await db['doctors'].find().to_list(length=10)
    with open('doctor_records.txt', 'w', encoding='utf-8') as f:
        for doc in docs:
            f.write(f"doctor_id: {doc['doctor_id']}\n")
            f.write(f"name: {doc['name']}\n")
            f.write(f"qualifications: {doc['qualifications']}\n")
            f.write(f"specialization: {doc['specialization']}\n")
            f.write(f"experience_years: {doc['experience_years']}\n")
            f.write(f"photo_path: {doc['photo_path']}\n")
            f.write(f"phone: {doc['phone']}\n")
            f.write(f"available_from: {doc['available_from']}\n")
            f.write(f"available_to: {doc['available_to']}\n")
            f.write('---\n')
    await db_manager.disconnect()

asyncio.run(check())