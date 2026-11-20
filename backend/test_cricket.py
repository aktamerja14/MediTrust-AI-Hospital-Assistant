import sys
sys.path.insert(0, 'F:\\AI & ML\\Gen AI\\MediTrust\\backend')
import asyncio
from app.database.connection import db_manager
from app.ai.rag.chain import RAGChain

async def test_cricket():
    await db_manager.connect()
    chain = RAGChain()
    query = "Tell me about cricket."
    response = await chain.generate(query)
    
    sources = [s['source'] for s in response.sources]
    print(f"Query: {query}")
    print(f"  Has context: {response.has_context}")
    print(f"  Sources: {sources}")
    print(f"  Answer: {response.answer}")
    print(f"  Doctor recs: {len(response.doctor_recommendations)}")

asyncio.run(test_cricket())