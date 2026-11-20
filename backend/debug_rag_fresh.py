import sys
sys.path.insert(0, 'F:\\AI & ML\\Gen AI\\MediTrust\\backend')
import asyncio

async def debug():
    from app.database.connection import db_manager
    from app.ai.rag.chain import RAGChain
    
    await db_manager.connect()
    chain = RAGChain()
    
    query = "What is diabetes and what are its symptoms?"
    response = await chain.generate(query)
    
    with open("rag_output_fresh.txt", "w", encoding="utf-8") as f:
        f.write(f"Has context: {response.has_context}\n")
        f.write(f"Sources: {len(response.sources)}\n")
        for s in response.sources:
            f.write(f"  Source: {s['source']}, score: {s['score']}\n")
        f.write(f"Answer: {response.answer}\n\n")
        f.write(f"Disclaimer: {response.disclaimer}\n\n")
        f.write(f"Doctor recs: {len(response.doctor_recommendations)}\n")

asyncio.run(debug())
print("Done")