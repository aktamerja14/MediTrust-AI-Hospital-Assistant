import asyncio
import json
import sys
import io
from pathlib import Path

# Fix Unicode output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

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
        token = response.json()['access_token']
        headers = {'Authorization': f'Bearer {token}'}
        
        queries = [
            "What is type 2 diabetes?",
            "What are the common symptoms of diabetes?",
            "What is diabetes? (in Gujarati)",
            "I have fever and headache. What should I do?",
            "Tell me about cricket.",
        ]
        
        for query in queries:
            print(f"\n{'='*60}")
            print(f"Query: {query}")
            print(f"{'='*60}")
            
            chat_payload = {'message': query, 'conversation_id': None}
            response = await client.post('/api/v1/chat/message', json=chat_payload, headers=headers)
            if response.status_code == 200:
                data = response.json()
                print(f"Has context: {data['has_context']}")
                print(f"Model: {data['model']}")
                print(f"Doctor recs: {len(data['doctor_recommendations'])}")
                print(f"\nResponse content:")
                print(data['assistant_message']['content'][:500])
                print("..." if len(data['assistant_message']['content']) > 500 else "")
            else:
                print(f"Error: {response.status_code}")
                print(response.text)
    
    await db_manager.disconnect()

asyncio.run(test_chat())