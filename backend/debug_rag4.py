import sys
sys.path.insert(0, 'F:\\AI & ML\\Gen AI\\MediTrust\\backend')
import os
print('Current working directory:', os.getcwd())
print('__file__ dir:', os.path.dirname(os.path.abspath(__file__)))

import asyncio
from app.core.config import settings
from app.ai.embeddings.embedding_factory import EmbeddingFactory
from app.ai.vectorstore.faiss_store import FAISSVectorStore
from pathlib import Path

async def debug():
    print('VECTOR_STORE_PATH from settings:', settings.VECTOR_STORE_PATH)
    print('Resolved path:', Path(settings.VECTOR_STORE_PATH).resolve())
    print('Path exists:', Path(settings.VECTOR_STORE_PATH).exists())
    
    # Create store like factory does
    embedding_provider = EmbeddingFactory.get_instance()
    dimension = embedding_provider.dimension
    print('Embedding dimension:', dimension)
    
    store = FAISSVectorStore(dimension=dimension, path=settings.VECTOR_STORE_PATH)
    print('Store path attribute:', store.path)
    print('Store path resolved:', Path(store.path).resolve())
    
    index_path = Path(store.path) / "index.faiss"
    meta_path = Path(store.path) / "metadata.json"
    id_map_path = Path(store.path) / "id_map.pkl"
    
    print('Index path:', index_path)
    print('Index path exists:', index_path.exists())
    print('Meta path exists:', meta_path.exists())
    print('ID map path exists:', id_map_path.exists())
    
    store.initialize()
    print('After initialize - ntotal:', store._index.ntotal)
    print('After initialize - docs:', len(store._documents))

asyncio.run(debug())