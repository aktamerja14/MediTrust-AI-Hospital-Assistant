import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').resolve()))
from dotenv import load_dotenv
load_dotenv(Path('.') / '.env')
from app.database.connection import DatabaseManager
from app.core.security.password import hash_password

async def test_chat():
    db_manager = DatabaseManager()
    await db_manager.connect()
    db = db_manager.db
    
    # Create a test user
    hashed = hash_password('test1234')
    await db['users'].update_one(
        {'email': 'testchat@test.com'},
        {'$set': {
            'email': 'testchat@test.com',
            'hashed_password': hashed,
            'role': 'patient',
            'is_active': True,
            'full_name': 'Test Chat User'
        }},
        upsert=True
    )
    print('Test user created/updated')
    
    await db_manager.disconnect()

asyncio.run(test_chat())