"""Admin API routes package."""
from app.api.routes.admin.routes import router as admin_router
from app.api.routes.admin.documents import router as documents_router

__all__ = ["admin_router", "documents_router"]