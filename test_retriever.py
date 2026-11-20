import sys
import os
import asyncio

backend_path = os.path.join(os.path.dirname(__file__), 'backend')
sys.path.insert(0, backend_path)
os.chdir(backend_path)

# Set test environment
os.environ["ENVIRONMENT"] = "testing"
os.environ["DATABASE_NAME"] = "meditrust_test"
os.environ["LLM_PROVIDER"] = "mock"
os.environ["LLM_MODEL"] = "mock-model"
os.environ["LLM_API_KEY"] = "mock-key"

from app.database.connection import db_manager
from app.ai.rag.retriever import Retriever


async def test():
    await db_manager.connect()
    db = db_manager.db
    
    # Test retriever
    retriever = Retriever()
    
    queries = [
        "I have a fever and headache",
        "I have a rash on my skin",
        "I have chest pain and high blood pressure",
        "I have severe headaches and numbness in my hands",
    ]
    
    for query in queries:
        results = retriever.retrieve(query, top_k=3)
        print(f"\nQuery: {query}")
        print(f"Results count: {len(results)}")
        for i, r in enumerate(results):
            print(f"  Result {i+1}: score={r.score:.4f}, source={r.document.metadata.get('source', 'Unknown')}, preview={r.document.content[:100]}...")
    
    await db_manager.disconnect()


if __name__ == "__main__":
    asyncio.run(test())