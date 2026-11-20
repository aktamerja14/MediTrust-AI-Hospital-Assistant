import sys
sys.path.insert(0, 'F:\\AI & ML\\Gen AI\\MediTrust\\backend')
import asyncio
from app.database.connection import db_manager
from app.ai.vectorstore.factory import VectorStoreFactory

async def debug():
    await db_manager.connect()
    store = VectorStoreFactory.get_instance()
    
    # Print all content from JSON sources
    for doc_id, doc in store._documents.items():
        source = doc.metadata.get('source', 'Unknown')
        if source in ['qa.json', 'hospital.json', 'gujarati.json', 'faq.json']:
            print(f"Doc: {doc_id}")
            print(f"Source: {source}")
            content = doc.content.replace('\n', ' ')
            print(f"Content: {content[:500]}...")
            print("---")

asyncio.run(debug())