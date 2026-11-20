import asyncio
from httpx import AsyncClient, ASGITransport
from app.main import create_application

async def test():
    app = create_application()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url='http://test') as ac:
        print('=== Legacy .jpg requests (should 404) ===')
        for photo in ['dr_deven.jpg', 'dr_rahul.jpg', 'dr_niddhi.jpg', 'dr_neeta.jpg', 'dr_shuresh.jpg', 'dr_mahesh_GyexvM8.jpg']:
            r = await ac.get(f'/doctor_photos/{photo}')
            print(f'  GET /doctor_photos/{photo} -> {r.status_code}, Content-Type: {r.headers.get("content-type")}')

asyncio.run(test())