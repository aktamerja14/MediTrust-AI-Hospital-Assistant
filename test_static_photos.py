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
        # Test static file serving for doctor photos
        for photo in ['dr_deven.svg', 'dr_rahul.svg', 'dr_niddhi.svg', 'dr_neeta.svg', 'dr_shuresh.svg', 'dr_mahesh_GyexvM8.svg']:
            response = await ac.get(f'/doctor_photos/{photo}')
            print(f'{photo}: Status {response.status_code}, Content-Type: {response.headers.get("content-type")}')
            if response.status_code != 200:
                print(f'  ERROR: {response.text[:200]}')

asyncio.run(test())