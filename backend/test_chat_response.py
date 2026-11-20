import asyncio
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').resolve()))
from dotenv import load_dotenv
load_dotenv(Path('.') / '.env')
from app.main import create_application
from app.database.connection import DatabaseManager
from app.api.dependencies import get_db as get_db_dep
from httpx import AsyncClient, ASGITransport

async def test_chat():
    app = create_application()
    db_manager = DatabaseManager()
    await db_manager.connect()
    db = db_manager.db
    app.dependency_overrides[get_db_dep] = lambda: db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url='http://test') as client:
        login_payload = {'email': 'testchat@test.com', 'password': 'test1234'}
        response = await client.post('/api/v1/auth/login', json=login_payload)
        print(f'Login Status: {response.status_code}')
        if response.status_code == 200:
            token = response.json()['access_token']
            headers = {'Authorization': f'Bearer {token}'}
            
            # Send a message
            chat_payload = {'message': 'What is diabetes and what are its symptoms?', 'conversation_id': None}
            response = await client.post('/api/v1/chat/message', json=chat_payload, headers=headers)
            print(f'Chat Status: {response.status_code}')
            print(json.dumps(response.json(), indent=2, default=str))
        await db_manager.disconnect()

asyncio.run(test_chat())