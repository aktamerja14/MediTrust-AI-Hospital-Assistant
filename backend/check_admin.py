import asyncio
from app.database.connection import db_manager
from app.core.security.password import hash_password, verify_password

async def check_admin():
    await db_manager.connect()
    db = db_manager.db
    users = await db.users.find({'role': 'admin'}).to_list(length=10)
    for user in users:
        print(f'Admin found: email={user["email"]}, role={user["role"]}, is_active={user["is_active"]}')
        print(f'Hash: {user["hashed_password"][:50]}...')
        test_passwords = ['admin123', 'password', 'Admin@123', 'meditrust', 'Admin@1234', 'MediTrust@123']
        for pwd in test_passwords:
            if verify_password(pwd, user['hashed_password']):
                print(f'  Password matches: {pwd}')
                break
        else:
            print('  No common password matched')
    await db_manager.disconnect()

asyncio.run(check_admin())