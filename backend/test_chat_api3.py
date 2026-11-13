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
            "email": "testtimestamp2@example.com",
            "password": "testpassword123"
        }
        response = await client.post("/api/v1/auth/signup", json=signup_payload)
        print(f"Signup Status: {response.status_code}")
        
        if response.status_code == 201:
            # Login
            response = await client.post("/api/v1/auth/login", json=signup_payload)
            token = response.json()["access_token"]
            headers = {"Authorization": f"Bearer {token}"}
            
            # Manually insert an old conversation (simulating old data)
            from bson import ObjectId
            from datetime import datetime, timezone
            old_conv = {
                "_id": ObjectId("6af5c6bc1996f510b7615e84"),
                "user_id": "6af75153e0f012fdc17c0c88",  # new user id
                "title": "Hello, test message",
                "messages": [
                    {"role": "user", "content": "Hello, test message", "timestamp": datetime(2026, 11, 12, 14, 1, 20, 749000)},
                    {"role": "assistant", "content": "I'm sorry, but I don't have enough information from the provided documents to answer this question.", "timestamp": datetime(2026, 11, 12, 14, 1, 32, 393000)}
                ],
                "created_at": datetime(2026, 11, 12, 14, 1, 20, 749000),
                "updated_at": datetime(2026, 11, 12, 14, 1, 32, 393000),
                "last_message_at": datetime(2026, 11, 12, 14, 1, 32, 393000)
            }
            await db['chat_history'].insert_one(old_conv)
            print("Inserted old conversation")
            
            # Get conversations
            response = await client.get("/api/v1/chat/conversations", headers=headers)
            print(f"\nConversations Status: {response.status_code}")
            print(f"Conversations Response: {response.json()}")

    await db_manager.disconnect()

asyncio.run(test_old_conversation())