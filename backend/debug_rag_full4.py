import sys
sys.path.insert(0, 'F:\\AI & ML\\Gen AI\\MediTrust\\backend')
import asyncio
from app.database.connection import db_manager
from app.ai.rag.chain import RAGChain

def safe_print(text):
    """Replace unicode chars that cp1252 can't handle"""
    return text.replace('\u2265', '>=').replace('\u2019', "'").replace('\u2011', '-').replace('\u2013', '-').replace('\u2014', '--').replace('\u2018', "'").replace('\u201c', '"').replace('\u201d', '"').replace('\u2026', '...')

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
        content_preview = safe_print(r.document.content[:200])
        print(f"  Result {i+1}: score={r.score:.4f}, source={source}")
        print(f"    Content preview: {content_preview}...")
        print()
    
    # Test full generation
    response = await chain.generate(query)
    print("=" * 60)
    print(f"Has context: {response.has_context}")
    print(f"Model: {response.model}")
    answer_preview = safe_print(response.answer[:1500])
    print(f"Answer: {answer_preview}")
    print()
    print(f"Disclaimer: {safe_print(response.disclaimer[:150])}...")
    print(f"Doctor recommendations: {len(response.doctor_recommendations)}")
    for dr in response.doctor_recommendations:
        name = safe_print(dr['name'])
        spec = safe_print(dr['specialization'])
        reason = safe_print(dr['match_reason'])
        print(f"  - {name} ({spec}) - {reason}")

asyncio.run(debug())