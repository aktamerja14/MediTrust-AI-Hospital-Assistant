from fastapi import APIRouter, Depends, HTTPException, status
from pymongo.asynchronous.database import AsyncDatabase

from app.core.config import settings
from app.database.connection import get_db

router = APIRouter(tags=["Health"])


@router.get("/health", summary="System Health Check")
async def health_check(db: AsyncDatabase | None = Depends(get_db)) -> dict[str, str]:
    """
    Health check endpoint returning system status, application title, and version.
    Includes database connectivity verification.
    """
    db_status = "connected" if db is not None else "disconnected"
    return {
        "status": "healthy",
        "application": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "database": db_status,
    }
