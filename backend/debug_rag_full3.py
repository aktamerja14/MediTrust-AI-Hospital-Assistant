import sys
sys.path.insert(0, 'F:\\AI & ML\\Gen AI\\MediTrust\\backend')
import asyncio
from app.database.connection import db_manager
from app.ai.rag.chain import RAGChain

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
        source = r.document.metadata.get('source')
        content_preview = r.document.content[:200].replace('\u2265', '>=').replace('\u2019', "'")
        print(f"  Result {i+1}: score={r.score:.4f}, source={source}")
        print(f"    Content preview: {content_preview}...")
        print()
    
    # Test full generation
    response = await chain.generate(query)
    print("=" * 60)
    print(f"Has context: {response.has_context}")
    print(f"Model: {response.model}")
    answer_preview = response.answer[:800].replace('\u2265', '>=').replace('\u2019', "'")
    print(f"Answer: {answer_preview}...")
    print(f"Disclaimer: {response.disclaimer[:100]}...")
    print(f"Doctor recommendations: {len(response.doctor_recommendations)}")
    for dr in response.doctor_recommendations:
        name = dr['name'].replace('\u2019', "'")
        spec = dr['specialization'].replace('\u2019', "'")
        reason = dr['match_reason'].replace('\u2019', "'")
        print(f"  - {name} ({spec}) - {reason}")

asyncio.run(debug())