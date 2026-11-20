"""Script to populate doctors collection from dataset."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.database.connection import db_manager
from app.core.config import settings


async def populate_doctors():
    """Populate doctors collection from the dataset."""
    await db_manager.connect()
    db = db_manager.db

    doctors_data = [
        {
            "doctor_id": "dr.deven",
            "name": "Dr. Deven Patel",
            "qualifications": "MBBS",
            "specialization": "General Physician (સામાન્ય રોગોની સારવાર)",
            "experience_years": 6,
            "photo_path": "doctor_photos/dr_deven.svg",
            "phone": "9700112232",
            "available_from": "10 a.m.",
            "available_to": "7:30 p.m.",
        },
        {
            "doctor_id": "dr.rahul",
            "name": "Dr. Rahul Maheta",
            "qualifications": "MBBS, MD (Neurology)",
            "specialization": "Neurologist (મગજ અને નર્વસ સિસ્ટમ)",
            "experience_years": 5,
            "photo_path": "doctor_photos/dr_rahul.svg",
            "phone": "9900112233",
            "available_from": "9 a.m.",
            "available_to": "6 p.m.",
        },
        {
            "doctor_id": "dr.niddhi",
            "name": "Dr. Nidhi Gopani",
            "qualifications": "MBBS, MD (Dermatology)",
            "specialization": "Dermatologist (ત્વચા, વાળ, નખ)",
            "experience_years": 7,
            "photo_path": "doctor_photos/dr_niddhi.svg",
            "phone": "9812345678",
            "available_from": "11 a.m.",
            "available_to": "7 p.m.",
        },
        {
            "doctor_id": "dr.nita",
            "name": "Dr. Nita Surani",
            "qualifications": "MBBS, MD (Orthopedics)",
            "specialization": "Orthopedics (હાડકાં અને સાંધાના રોગ)",
            "experience_years": 5,
            "photo_path": "doctor_photos/dr_neeta.svg",
            "phone": "9898123456",
            "available_from": "9:30 a.m.",
            "available_to": "5:30 p.m.",
        },
        {
            "doctor_id": "dr.suresh",
            "name": "Dr. Suresh Patel",
            "qualifications": "MBBS, MS (Gynecology)",
            "specialization": "Gynecologist (મહિલાઓના રોગ / ગર્ભાવસ્થા)",
            "experience_years": 8,
            "photo_path": "doctor_photos/dr_shuresh.svg",
            "phone": "9823456712",
            "available_from": "5 a.m.",
            "available_to": "10 p.m.",
        },
        {
            "doctor_id": "dr.mahesh",
            "name": "Dr. Mahesh Ramani",
            "qualifications": "MBBS, MD (Cardiology)",
            "specialization": "Cardiologist (હૃદયના રોગ)",
            "experience_years": 12,
            "photo_path": "doctor_photos/dr_mahesh_GyexvM8.svg",
            "phone": "9876543210",
            "available_from": "5 a.m.",
            "available_to": "11 p.m.",
        },
    ]

    created_count = 0
    skipped_count = 0

    for doctor in doctors_data:
        existing = await db["doctors"].find_one({"doctor_id": doctor["doctor_id"]})
        if existing:
            print(f"Skipped (already exists): {doctor['doctor_id']} - {doctor['name']}")
            skipped_count += 1
            continue

        await db["doctors"].insert_one(doctor)
        print(f"Created: {doctor['doctor_id']} - {doctor['name']}")
        created_count += 1

    # Create index on doctor_id for uniqueness
    await db["doctors"].create_index("doctor_id", unique=True)

    print(f"\nPopulation complete:")
    print(f"  Created: {created_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"  Total in dataset: {len(doctors_data)}")

    await db_manager.disconnect()


if __name__ == "__main__":
    asyncio.run(populate_doctors())