import asyncio
import os
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))

from dotenv import load_dotenv
load_dotenv(BACKEND_DIR / ".env")

from app.main import create_application
from app.database.connection import DatabaseManager, get_db
from app.api.dependencies import get_db as get_db_dep
from httpx import AsyncClient, ASGITransport

async def test_old_conversation():
    app = create_application()
    db_manager = DatabaseManager()
    await db_manager.connect()
    db = db_manager.db

    app.dependency_overrides[get_db_dep] = lambda: db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Login with existing user who has old conversations
        signup_payload = {
            "email": "testtimestamp4@example.com",
            "password": "testpassword123"
        }
        response = await client.post("/api/v1/auth/signup", json=signup_payload)
        print(f"Signup Status: {response.status_code}")
        
        if response.status_code == 201:
            # Login
            response = await client.post("/api/v1/auth/login", json=signup_payload)
            print(f"Login Response: {response.json()}")
            token = response.json()["access_token"]
            headers = {"Authorization": f"Bearer {token}"}
            
            # Get the new user's ObjectId
            new_user = await db['users'].find_one({"email": "testtimestamp4@example.com"})
            print(f"New user ID: {new_user['_id']}")
            
            # Get existing old conversations for the original user
            from bson import ObjectId
            old_convs = await db['chat_history'].find({"user_id": "6af5c6b01996f510b7615e83"}).to_list(length=5)
            print(f"Found {len(old_convs)} old conversations")
            
            # Copy one to new user with correct user_id
            if old_convs:
                old_conv = old_convs[0]
                old_conv['user_id'] = new_user['_id']
                old_conv['_id'] = ObjectId()
                await db['chat_history'].insert_one(old_conv)
                print("Inserted old conversation for new user")
                print(f"Old conversation timestamps:")
                print(f"  created_at: {old_conv.get('created_at')}")
                print(f"  updated_at: {old_conv.get('updated_at')}")
                print(f"  last_message_at: {old_conv.get('last_message_at')}")
                for i, msg in enumerate(old_conv.get('messages', [])):
                    print(f"  Message {i}: timestamp={msg.get('timestamp')}")
            
            # Get conversations
            response = await client.get("/api/v1/chat/conversations", headers=headers)
            print(f"\nConversations Status: {response.status_code}")
            print(f"Conversations Response: {response.json()}")

    await db_manager.disconnect()

asyncio.run(test_old_conversation())