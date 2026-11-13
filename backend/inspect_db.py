import asyncio
from app.database.connection import db_manager
from app.core.config import settings

async def inspect():
    await db_manager.connect()
    db = db_manager.db
    cursor = db['chat_history'].find({}).limit(5)
    async for doc in cursor:
        print('---')
        print(f'_id: {doc["_id"]}')
        print(f'user_id: {doc.get("user_id")}')
        print(f'title: {doc.get("title")}')
        print(f'created_at: {doc.get("created_at")} (type: {type(doc.get("created_at"))})')
        print(f'updated_at: {doc.get("updated_at")} (type: {type(doc.get("updated_at"))})')
        print(f'last_message_at: {doc.get("last_message_at")} (type: {type(doc.get("last_message_at"))})')
        messages = doc.get('messages', [])
        if messages:
            last_msg = messages[-1]
            print(f'Last message timestamp: {last_msg.get("timestamp")} (type: {type(last_msg.get("timestamp"))})')
            print(f'Last message content: {last_msg.get("content", "")[:50]}...')
        else:
            print('No messages')
    await db_manager.disconnect()

asyncio.run(inspect())