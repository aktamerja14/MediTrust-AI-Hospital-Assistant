import sys
sys.path.insert(0, 'F:\\AI & ML\\Gen AI\\MediTrust\\backend')
import asyncio
from app.database.connection import db_manager
from app.ai.vectorstore.factory import VectorStoreFactory
from app.ai.embeddings.embedding_factory import EmbeddingFactory

async def debug():
    await db_manager.connect()
    store = VectorStoreFactory.get_instance()
    embedding_provider = EmbeddingFactory.get_instance()
    
    query = "What is diabetes and what are its symptoms?"
    query_embedding = embedding_provider.embed_query(query)
    
    # Search with higher k to see more results
    results = store.search(query_embedding, top_k=20)
    print(f"Total results: {len(results)}")
    for i, r in enumerate(results):
        source = r.document.metadata.get('source', 'Unknown')
        print(f"  Result {i+1}: score={r.score:.4f}, source={source}")
        print(f"    Content preview: {r.document.content[:150]}...")
        print()

asyncio.run(debug())