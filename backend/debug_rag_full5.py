import sys
sys.path.insert(0, 'F:\\AI & ML\\Gen AI\\MediTrust\\backend')
import asyncio
from app.database.connection import db_manager
from app.ai.rag.chain import RAGChain

async def debug():
    await db_manager.connect()
    chain = RAGChain()
    
    query = "What is diabetes and what are its symptoms?"
    
    # Test retrieval directly
    results = chain.retriever.retrieve(query, top_k=4)
    
    # Test full generation
    response = await chain.generate(query)
    
    # Write to file instead of printing
    with open("rag_output.txt", "w", encoding="utf-8") as f:
        f.write(f"Query: {query}\n")
        f.write("=" * 60 + "\n")
        f.write(f"Retrieved {len(results)} results:\n")
        for i, r in enumerate(results):
            source = r.document.metadata.get('source')
            content_preview = r.document.content[:200]
            f.write(f"  Result {i+1}: score={r.score:.4f}, source={source}\n")
            f.write(f"    Content preview: {content_preview}...\n\n")
        
        f.write("=" * 60 + "\n")
        f.write(f"Has context: {response.has_context}\n")
        f.write(f"Model: {response.model}\n")
        f.write(f"Answer: {response.answer}\n\n")
        f.write(f"Disclaimer: {response.disclaimer}\n\n")
        f.write(f"Doctor recommendations: {len(response.doctor_recommendations)}\n")
        for dr in response.doctor_recommendations:
            f.write(f"  - {dr['name']} ({dr['specialization']}) - {dr['match_reason']}\n")
    
    print("Output written to rag_output.txt")

asyncio.run(debug())