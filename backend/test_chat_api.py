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
        # Login with existing user
        login_payload = {
            "email": "radha123@gmail.com",
            "password": "radha123"  # Common test password
        }
        response = await client.post("/api/v1/auth/login", json=login_payload)
        print(f"Login Status: {response.status_code}")
        print(f"Login Response: {response.json()}")
        
        if response.status_code == 200:
            token = response.json()["access_token"]
            headers = {"Authorization": f"Bearer {token}"}
            
            # Get conversations
            response = await client.get("/api/v1/chat/conversations", headers=headers)
            print(f"\nConversations Status: {response.status_code}")
            print(f"Conversations Response: {response.json()}")
        else:
            # Try different password
            print("Trying different passwords...")
            for pwd in ["password123", "radha123@gmail.com", "12345678", "test1234"]:
                login_payload = {"email": "radha123@gmail.com", "password": pwd}
                response = await client.post("/api/v1/auth/login", json=login_payload)
                print(f"  Password '{pwd}': {response.status_code}")
                if response.status_code == 200:
                    print(f"  Success! Token: {response.json()['access_token'][:20]}...")
                    break

    await db_manager.disconnect()

asyncio.run(test_conversations())