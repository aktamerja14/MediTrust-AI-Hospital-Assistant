import sys
sys.path.insert(0, 'F:\\AI & ML\\Gen AI\\MediTrust\\backend')
import asyncio
from app.database.connection import db_manager
from app.ai.vectorstore.factory import VectorStoreFactory

async def debug():
    await db_manager.connect()
    store = VectorStoreFactory.get_instance()
    
    # Collect unique sources
    sources = {}
    for doc_id, doc in store._documents.items():
        source = doc.metadata.get('source', 'Unknown')
        if source not in sources:
            sources[source] = 0
        sources[source] += 1
    
    print("Unique sources and chunk counts:")
    for source, count in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"  {source}: {count} chunks")

asyncio.run(debug())