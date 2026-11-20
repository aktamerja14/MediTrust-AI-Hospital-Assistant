import asyncio
from httpx import AsyncClient, ASGITransport
from app.main import create_application

async def test():
    app = create_application()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url='http://test') as ac:
        # Test public doctors endpoint
        resp = await ac.get('/api/v1/doctors/public')
        print('=== /api/v1/doctors/public ===')
        print(f'Status: {resp.status_code}')
        for d in resp.json()['doctors']:
            print(f"  {d['doctor_id']}: photo_path={d['photo_path']}")
        
        print()
        print('=== Static file serving (.svg) ===')
        for photo in ['dr_deven.svg', 'dr_rahul.svg', 'dr_niddhi.svg', 'dr_neeta.svg', 'dr_shuresh.svg', 'dr_mahesh_GyexvM8.svg']:
            r = await ac.get(f'/doctor_photos/{photo}')
            print(f'  GET /doctor_photos/{photo} -> {r.status_code}, Content-Type: {r.headers.get("content-type")}')
        
        print()
        print('=== Legacy .jpg requests (should 404) ===')
        for photo in ['dr_deven.jpg', 'dr_rahul.jpg', 'dr_niddhi.jpg', 'dr_neeta.jpg', 'dr_shuresh.jpg', 'dr_mahesh_GyexvM8.jpg']:
            r = await ac.get(f'/doctor_photos/{photo}')
            print(f'  GET /doctor_photos/{photo} -> {r.status_code}, Content-Type: {r.headers.get("content-type")}')

asyncio.run(test())