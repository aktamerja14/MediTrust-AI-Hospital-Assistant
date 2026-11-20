import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').resolve()))
from dotenv import load_dotenv
load_dotenv(Path('.') / '.env')
from app.ai.vectorstore.factory import VectorStoreFactory
from app.ai.embeddings.embedding_factory import EmbeddingFactory

store = VectorStoreFactory.get_instance()
print('Index size:', store.index.ntotal if hasattr(store, 'index') and store.index else 'N/A')

embedding_provider = EmbeddingFactory.get_instance()
query_emb = embedding_provider.embed_query('diabetes education what is diabetes')
print('Embedding type:', type(query_emb))

results = store.search(query_emb, top_k=20)
print('Total results:', len(results))
for i, r in enumerate(results):
    source = r.document.metadata.get('source', 'Unknown')
    print(f'  {i+1}: score={r.score:.4f}, source={source}, preview={r.document.content[:100]}...')