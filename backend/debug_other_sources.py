import sys
sys.path.insert(0, 'F:\\AI & ML\\Gen AI\\MediTrust\\backend')
import asyncio
from app.database.connection import db_manager
from app.ai.vectorstore.factory import VectorStoreFactory

async def debug():
    await db_manager.connect()
    store = VectorStoreFactory.get_instance()
    
    # Print all content from other sources
    for doc_id, doc in store._documents.items():
        source = doc.metadata.get('source', 'Unknown')
        if source not in ['symptom-based-disease-prediction-v1.txt', 'medical_medicine_dataset.txt']:
            print(f"Doc: {doc_id}")
            print(f"Source: {source}")
            print(f"Content: {doc.content[:500]}...")
            print("---")

asyncio.run(debug())