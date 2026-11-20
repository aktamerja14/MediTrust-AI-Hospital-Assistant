import sys
import os
import asyncio

backend_path = os.path.join(os.path.dirname(__file__), 'backend')
sys.path.insert(0, backend_path)
os.chdir(backend_path)

# Set test environment
os.environ["ENVIRONMENT"] = "testing"
os.environ["DATABASE_NAME"] = "meditrust_test"
os.environ["LLM_PROVIDER"] = "mock"
os.environ["LLM_MODEL"] = "mock-model"
os.environ["LLM_API_KEY"] = "mock-key"

from app.database.connection import db_manager
from app.ai.rag.chain import RAGChain


async def test():
    await db_manager.connect()
    db = db_manager.db
    
    # Populate doctors
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
    
    for doctor in doctors_data:
        await db["doctors"].update_one(
            {"doctor_id": doctor["doctor_id"]},
            {"$set": doctor},
            upsert=True
        )
    
    await db["doctors"].create_index("doctor_id", unique=True)
    
    # Test RAG chain
    rag_chain = RAGChain()
    
    queries = [
        "I have a fever and headache",
        "I have a rash on my skin",
        "I have chest pain and high blood pressure",
        "I have severe headaches and numbness in my hands",
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        try:
            response = await rag_chain.generate(query=query, db=db)
            print(f"  Answer: {response.answer[:200]}...")
            print(f"  Has context: {response.has_context}")
            print(f"  Doctor recommendations: {len(response.doctor_recommendations or [])}")
            for rec in (response.doctor_recommendations or []):
                print(f"    - {rec['name']} ({rec['doctor_id']})")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    await db_manager.disconnect()


if __name__ == "__main__":
    asyncio.run(test())