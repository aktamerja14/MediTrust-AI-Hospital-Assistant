import sys
sys.path.insert(0, 'F:\\AI & ML\\Gen AI\\MediTrust\\backend')
import asyncio
from app.database.connection import db_manager
from app.ai.rag.chain import RAGChain

async def debug():
    await db_manager.connect()
    chain = RAGChain()
    
    query = "What is diabetes and what are its symptoms?"
    
    # Test with top_k=3 to only get diabetes_education chunks
    results = chain.retriever.retrieve(query, top_k=3)
    
    print("Retrieved results:")
    for i, r in enumerate(results):
        source = r.document.metadata.get('source')
        print(f"  Result {i+1}: score={r.score:.4f}, source={source}")
    
    # Test full generation with only top 3
    response = await chain.generate(query, top_k=3)
    
    with open("rag_output_top3.txt", "w", encoding="utf-8") as f:
        f.write(f"Has context: {response.has_context}\n")
        f.write(f"Answer: {response.answer}\n")
        f.write(f"Disclaimer: {response.disclaimer}\n")
        f.write(f"Doctor recs: {len(response.doctor_recommendations)}\n")
    
    print("Output written to rag_output_top3.txt")

asyncio.run(debug())