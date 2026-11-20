import asyncio
import sys
sys.stdout.reconfigure(encoding="utf-8")
from app.ai.rag.retriever import Retriever

async def test():
    retriever = Retriever()
    results = retriever.retrieve_with_scores("What is diabetes?", top_k=5, min_score=0.55)
    print("Results with min_score=0.55:", len(results))
    for r in results:
        source = r.document.metadata.get("source", "N/A")
        print("  Score: {:.4f}, Source: {}".format(r.score, source))
    
    results2 = retriever.retrieve_with_scores("What is diabetes?", top_k=5, min_score=0.3)
    print("Results with min_score=0.3:", len(results2))
    for r in results2:
        source = r.document.metadata.get("source", "N/A")
        print("  Score: {:.4f}, Source: {}".format(r.score, source))

asyncio.run(test())