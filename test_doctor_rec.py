import sys
import os
import asyncio
import json

backend_path = os.path.join(os.path.dirname(__file__), 'backend')
sys.path.insert(0, backend_path)
os.chdir(backend_path)

from app.database.connection import db_manager
from app.ai.rag.chain import _detect_specialization, _get_doctor_recommendations


def safe_print(obj):
    """Print object safely handling Unicode."""
    try:
        print(json.dumps(obj, ensure_ascii=False, indent=2))
    except:
        print(str(obj).encode('ascii', 'replace').decode('ascii'))


async def test():
    await db_manager.connect()
    db = db_manager.db
    
    # Test specialization detection
    query = "I have a fever and headache"
    specs = _detect_specialization(query)
    safe_print({"query": query, "specializations": specs})
    
    # Test doctor recommendations
    recs = await _get_doctor_recommendations(specs, db=db)
    safe_print({"query": query, "recommendations": recs})
    
    # Test with skin query
    query2 = "I have a rash on my skin"
    specs2 = _detect_specialization(query2)
    safe_print({"query": query2, "specializations": specs2})
    recs2 = await _get_doctor_recommendations(specs2, db=db)
    safe_print({"query": query2, "recommendations": recs2})
    
    # Test with heart query
    query3 = "I have chest pain and high blood pressure"
    specs3 = _detect_specialization(query3)
    safe_print({"query": query3, "specializations": specs3})
    recs3 = await _get_doctor_recommendations(specs3, db=db)
    safe_print({"query": query3, "recommendations": recs3})
    
    # Test with neurology query
    query4 = "I have severe headaches and numbness in my hands"
    specs4 = _detect_specialization(query4)
    safe_print({"query": query4, "specializations": specs4})
    recs4 = await _get_doctor_recommendations(specs4, db=db)
    safe_print({"query": query4, "recommendations": recs4})
    
    await db_manager.disconnect()


if __name__ == "__main__":
    asyncio.run(test())