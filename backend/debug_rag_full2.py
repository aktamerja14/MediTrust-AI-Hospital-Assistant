import sys
sys.path.insert(0, 'F:\\AI & ML\\Gen AI\\MediTrust\\backend')
import asyncio
from app.database.connection import db_manager
from app.ai.rag.chain import RAGChain
from app.ai.rag.prompts import build_rag_prompt

async def debug():
    await db_manager.connect()
    chain = RAGChain()
    
    query = "What is diabetes and what are its symptoms?"
    print(f"Query: {query}")
    print("=" * 60)
    
    # Test retrieval directly
    results = chain.retriever.retrieve(query, top_k=4)
    print(f"Retrieved {len(results)} results:")
    for i, r in enumerate(results):
        print(f"  Result {i+1}: score={r.score:.4f}, source={r.document.metadata.get('source')}")
        print(f"    FULL Content: {r.document.content}")
        print()
    
    # Check what prompt is built
    system_prompt, user_prompt = build_rag_prompt(query, results, None)
    print("=" * 60)
    print("SYSTEM PROMPT:")
    print(system_prompt)
    print()
    print("USER PROMPT:")
    print(user_prompt)

asyncio.run(debug())