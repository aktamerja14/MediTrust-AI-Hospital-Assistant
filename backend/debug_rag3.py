import sys
sys.path.insert(0, 'F:\\AI & ML\\Gen AI\\MediTrust\\backend')
import asyncio
import faiss
import json
import pickle
from pathlib import Path

async def debug():
    path = Path("F:/AI & ML/Gen AI/MediTrust/backend/data/vector_store")
    index_path = path / "index.faiss"
    meta_path = path / "metadata.json"
    id_map_path = path / "id_map.pkl"
    
    print('Index path:', index_path)
    print('Index path exists:', index_path.exists())
    print('Meta path exists:', meta_path.exists())
    print('ID map path exists:', id_map_path.exists())
    
    # Try loading directly
    index = faiss.read_index(str(index_path))
    print('FAISS index ntotal:', index.ntotal)
    print('FAISS index dimension:', index.d)
    
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    print('Metadata keys:', len(meta))
    if meta:
        first_key = list(meta.keys())[0]
        print('First doc:', first_key)
        first_doc = meta[first_key]
        print('First doc content preview:', first_doc.get('content', '')[:200])
        print('First doc metadata:', first_doc.get('metadata', {}))
    
    with open(id_map_path, 'rb') as f:
        id_map = pickle.load(f)
    print('ID map length:', len(id_map))
    print('ID map:', list(id_map.items())[:5])

asyncio.run(debug())