import asyncio
from app.database.connection import db_manager
from bson import ObjectId
from datetime import timezone

async def check():
    await db_manager.connect()
    db = db_manager.db
    cursor = db['chat_history'].find({})
    async for doc in cursor:
        oid = doc['_id']
        gen_time = oid.generation_time.replace(tzinfo=None)  # Make naive for comparison
        created = doc.get('created_at')
        last_msg = doc.get('last_message_at')
        print(f'Conv {oid}:')
        print(f'  ObjectId time (UTC): {gen_time}')
        print(f'  created_at: {created}')
        print(f'  last_message_at: {last_msg}')
        if created:
            diff = abs((gen_time - created).total_seconds())
            print(f'  Match created: {diff < 2} (diff: {diff}s)')
        if last_msg:
            diff = abs((gen_time - last_msg).total_seconds())
            print(f'  Match last_msg: {diff < 2} (diff: {diff}s)')
    await db_manager.disconnect()

asyncio.run(check())