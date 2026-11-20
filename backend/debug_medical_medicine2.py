import sys
sys.path.insert(0, 'F:\\AI & ML\\Gen AI\\MediTrust\\backend')
import asyncio
from app.database.connection import db_manager
from app.ai.vectorstore.factory import VectorStoreFactory

async def debug():
    await db_manager.connect()
    store = VectorStoreFactory.get_instance()
    
    # Print all content from medical_medicine_dataset.txt
    for doc_id, doc in store._documents.items():
        source = doc.document.metadata.get('source', 'Unknown') if hasattr(doc, 'document') else doc.metadata.get('source', 'Unknown')
        if source == 'medical_medicine_dataset.txt':
            content = doc.content if hasattr(doc, 'content') else doc.content
            print(f"Doc: {doc_id}")
            print(f"Content: {content[:500]}...")
            print("---")

asyncio.run(debug())