import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').resolve()))
from dotenv import load_dotenv
load_dotenv(Path('.') / '.env')
from app.ai.rag.retriever import Retriever

async def test_retriever():
    retriever = Retriever()
    results = retriever.retrieve('I have fever and headache. What should I do?', top_k=10)
    for i, r in enumerate(results):
        source = r.document.metadata.get('source', 'Unknown')
        print(f'Result {i+1}: score={r.score:.4f}, source={source}')
        print(f'  Preview: {r.document.content[:200]}...')
        print()

asyncio.run(test_retriever())