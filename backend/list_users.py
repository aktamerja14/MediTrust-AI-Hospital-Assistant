import asyncio
from app.database.connection import db_manager

async def list_users():
    await db_manager.connect()
    db = db_manager.db
    users = await db.users.find({}).to_list(length=20)
    for user in users:
        print(f'email={user["email"]}, role={user["role"]}, is_active={user["is_active"]}, id={user["_id"]}')
    await db_manager.disconnect()

asyncio.run(list_users())