import sys
sys.path.insert(0, 'F:\\AI & ML\\Gen AI\\MediTrust\\backend')
import asyncio
from app.database.connection import db_manager
from app.ai.vectorstore.factory import VectorStoreFactory

async def debug():
    await db_manager.connect()
    store = VectorStoreFactory.get_instance()
    
    # Search all documents for diabetes content
    diabetes_docs = []
    for doc_id, doc in store._documents.items():
        content = doc.content.lower()
        if 'diabetes' in content:
            source = doc.metadata.get('source', 'Unknown')
            diabetes_docs.append((doc_id, source, doc.content[:300]))
    
    print(f"Found {len(diabetes_docs)} chunks mentioning diabetes:")
    for doc_id, source, content in diabetes_docs[:20]:
        print(f"  Doc: {doc_id}")
        print(f"  Source: {source}")
        print(f"  Content: {content}...")
        print()

asyncio.run(debug())