import asyncio
from app.database.connection import db_manager

async def check():
    await db_manager.connect()
    db = db_manager.db
    users = await db.users.find({}).to_list(length=20)
    for user in users:
        print(f'email={user["email"]}, id={user["_id"]}')
    await db_manager.disconnect()

asyncio.run(check())