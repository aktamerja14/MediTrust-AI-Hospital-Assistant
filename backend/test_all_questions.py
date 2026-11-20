import sys
sys.path.insert(0, 'F:\\AI & ML\\Gen AI\\MediTrust\\backend')
import asyncio
from app.database.connection import db_manager
from app.ai.rag.chain import RAGChain

async def test_question(query):
    await db_manager.connect()
    chain = RAGChain()
    response = await chain.generate(query)
    
    sources = [s['source'] for s in response.sources]
    print(f"Query: {query}")
    print(f"  Has context: {response.has_context}")
    print(f"  Sources: {sources}")
    print(f"  Answer preview: {response.answer[:200]}...")
    print(f"  Doctor recs: {len(response.doctor_recommendations)}")
    print()

async def main():
    questions = [
        "What is diabetes and what are its symptoms?",
        "What is type 2 diabetes?",
        "What are common symptoms of diabetes?",
        "What is diabetes in Gujarati?",
        "Tell me about cricket.",
    ]
    
    for q in questions:
        await test_question(q)

asyncio.run(main())