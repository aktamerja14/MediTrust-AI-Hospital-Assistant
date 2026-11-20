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
    results = chain.retriever.retrieve(query, top_k=4)
    
    print("Retrieved results:")
    for i, r in enumerate(results):
        source = r.document.metadata.get('source')
        print(f"  Result {i+1}: score={r.score:.4f}, source={source}")
        print(f"    Content: {r.document.content[:300]}...")
        print()
    
    system_prompt, user_prompt = build_rag_prompt(query, results, None)
    
    with open("prompt_output.txt", "w", encoding="utf-8") as f:
        f.write("SYSTEM PROMPT:\n")
        f.write(system_prompt + "\n\n")
        f.write("USER PROMPT:\n")
        f.write(user_prompt)
    
    print("Prompt written to prompt_output.txt")

asyncio.run(debug())