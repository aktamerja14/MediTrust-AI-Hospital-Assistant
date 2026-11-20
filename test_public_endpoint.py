import asyncio
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from httpx import AsyncClient, ASGITransport
from app.main import create_application

async def test():
    app = create_application()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url='http://test') as ac:
        response = await ac.get('/api/v1/doctors/public')
        print(f'Status: {response.status_code}')
        data = response.json()
        print(f'Found {len(data["doctors"])} doctors')
        for d in data['doctors']:
            print(f'  - {d["name"]} ({d["doctor_id"]})')

asyncio.run(test())