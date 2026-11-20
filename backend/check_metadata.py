import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').resolve()))
from dotenv import load_dotenv
load_dotenv(Path('.') / '.env')
from app.ai.vectorstore.factory import VectorStoreFactory

store = VectorStoreFactory.get_instance()
if hasattr(store, 'metadata') and store.metadata:
    print('Metadata count:', len(store.metadata))
    sources = set()
    for m in store.metadata:
        src = m.get('source', 'Unknown')
        sources.add(src)
    print('Unique sources:', sources)
    
    for i, m in enumerate(store.metadata[:10]):
        source = m.get('source', 'Unknown')
        chunk_idx = m.get('chunk_index', 'N/A')
        content = m.get('content', '')[:80]
        print(f'  {i}: source={source}, chunk_index={chunk_idx}, preview={content}...')
else:
    print('No metadata available')