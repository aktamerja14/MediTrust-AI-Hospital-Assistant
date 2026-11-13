import asyncio
import os
import sys
from pathlib import Path

# Add backend to path
BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))

# Load .env before importing settings
from dotenv import load_dotenv
load_dotenv(BACKEND_DIR / ".env")

from app.main import create_application
from app.database.connection import DatabaseManager, get_db
from app.api.dependencies import get_db as get_db_dep
from httpx import AsyncClient, ASGITransport

async def test_conversations():
    app = create_application()
    db_manager = DatabaseManager()
    await db_manager.connect()
    db = db_manager.db

    app.dependency_overrides[get_db_dep] = lambda: db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Create a new test user
        signup_payload = {
            "email": "testtimestamp@example.com",
            "password": "testpassword123"
        }
        response = await client.post("/api/v1/auth/signup", json=signup_payload)
        print(f"Signup Status: {response.status_code}")
        print(f"Signup Response: {response.json()}")
        
        if response.status_code == 201:
            # Login
            response = await client.post("/api/v1/auth/login", json=signup_payload)
            print(f"\nLogin Status: {response.status_code}")
            print(f"Login Response: {response.json()}")
            
            if response.status_code == 200:
                token = response.json()["access_token"]
                headers = {"Authorization": f"Bearer {token}"}
                
                # Send a message to create a conversation
                response = await client.post("/api/v1/chat/message", json={
                    "message": "Hello, test message for timestamp debugging"
                }, headers=headers)
                print(f"\nSend Message Status: {response.status_code}")
                print(f"Send Message Response: {response.json()}")
                
                # Get conversations
                response = await client.get("/api/v1/chat/conversations", headers=headers)
                print(f"\nConversations Status: {response.status_code}")
                print(f"Conversations Response: {response.json()}")

    await db_manager.disconnect()

asyncio.run(test_conversations())