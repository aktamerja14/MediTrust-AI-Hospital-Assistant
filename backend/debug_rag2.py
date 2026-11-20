import sys
sys.path.insert(0, 'F:\\AI & ML\\Gen AI\\MediTrust\\backend')
import asyncio
from app.database.connection import db_manager
from app.ai.vectorstore.factory import VectorStoreFactory
from app.ai.embeddings.embedding_factory import EmbeddingFactory

async def debug():
    await db_manager.connect()
    store = VectorStoreFactory.get_instance()
    
    # Check if initialized
    print('_initialized:', store._initialized)
    
    # Force initialize
    store.initialize()
    print('After initialize:')
    print('_initialized:', store._initialized)
    print('Vector store stats:', store.get_stats())
    print('Total vectors:', store._index.ntotal if store._index else 0)
    print('Tracked documents:', len(store._documents))
    
    # Show first 10 documents
    for i, (doc_id, doc) in enumerate(list(store._documents.items())[:10]):
        source = doc.metadata.get('source', 'Unknown')
        content_preview = doc.content[:100].replace('\n', ' ')
        print(f'Doc {i}: id={doc_id}, source={source}, content_preview={content_preview}...')

asyncio.run(debug())