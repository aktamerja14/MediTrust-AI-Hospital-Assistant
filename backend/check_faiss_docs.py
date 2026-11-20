import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').resolve()))
from dotenv import load_dotenv
load_dotenv(Path('.') / '.env')
from app.ai.vectorstore.factory import VectorStoreFactory

store = VectorStoreFactory.get_instance()
store.initialize()

print('Total vectors:', store._index.ntotal)
print('Tracked documents:', len(store._documents))

sources = set()
for doc_id, doc in store._documents.items():
    src = doc.metadata.get('source', 'Unknown')
    sources.add(src)

print('Unique sources:', sources)

# Check for diabetes_education
for doc_id, doc in store._documents.items():
    src = doc.metadata.get('source', 'Unknown')
    if 'diabetes' in src.lower() or 'education' in src.lower():
        print(f'Found diabetes doc: {doc_id}, source={src}, preview={doc.content[:100]}...')
        break
else:
    print('No diabetes_education document found in FAISS store')

# Show first few docs
for i, (doc_id, doc) in enumerate(list(store._documents.items())[:10]):
    src = doc.metadata.get('source', 'Unknown')
    chunk = doc.metadata.get('chunk_index', 'N/A')
    print(f'  {i}: id={doc_id}, source={src}, chunk={chunk}, preview={doc.content[:80]}...')