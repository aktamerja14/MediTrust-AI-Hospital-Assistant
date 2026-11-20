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
    
    # Search with filter for medical_medicine_dataset.txt
    results = store.search(query_embedding, top_k=50)
    
    print("Results from medical_medicine_dataset.txt:")
    for i, r in enumerate(results):
        source = r.document.metadata.get('source', 'Unknown')
        if source == 'medical_medicine_dataset.txt':
            print(f"  Result {i+1}: score={r.score:.4f}, source={source}")
            print(f"    Content preview: {r.document.content[:200]}...")
            print()

asyncio.run(debug())