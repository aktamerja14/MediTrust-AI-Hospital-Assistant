import sys
sys.path.insert(0, 'F:\\AI & ML\\Gen AI\\MediTrust\\backend')
import asyncio
from app.ai.factory import ProviderFactory
from app.ai.providers.base import LLMMessage

async def debug():
    provider = ProviderFactory.get_instance()
    print(f"Provider: {provider.__class__.__name__}")
    print(f"Model: {provider.model}")
    
    system_prompt = "You are MediTrust AI Health Assistant. Answer based ONLY on the provided medical context."
    
    user_prompt = """Medical context:
Diabetes mellitus is a chronic metabolic disorder characterized by elevated blood glucose levels. Common symptoms include frequent urination, excessive thirst, unexplained weight loss, fatigue, blurred vision, slow-healing cuts, and tingling in hands/feet.

User question: What is diabetes and what are its symptoms?

Answer based ONLY on the provided medical context."""
    
    messages = [LLMMessage(role="user", content=user_prompt)]
    response = provider.generate_response(messages=messages, system_prompt=system_prompt)
    
    with open("llm_output.txt", "w", encoding="utf-8") as f:
        f.write(response.content)
    
    print("Output written to llm_output.txt")

asyncio.run(debug())