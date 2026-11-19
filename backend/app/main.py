from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.api.routes import health
from app.api.routes.auth import router as auth_router
from app.api.routes.admin import admin_router, documents_router
from app.api.routes.chat import chat_router
from app.api.routes.doctors import router as doctors_router
from app.database.connection import lifespan


def create_application() -> FastAPI:
    """Factory function to configure and initialize the FastAPI application."""
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description="MediTrust Hospital Management & Clinical Services REST API",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Configure Cross-Origin Resource Sharing (CORS)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Static file serving for doctor photos
    doctor_photos_path = Path(__file__).resolve().parent.parent / "doctor_photos"
    doctor_photos_path.mkdir(exist_ok=True)
    app.mount("/doctor_photos", StaticFiles(directory=str(doctor_photos_path)), name="doctor_photos")

    # Register API Routers
    app.include_router(health.router, prefix=settings.API_V1_STR)
    app.include_router(auth_router, prefix=settings.API_V1_STR)
    app.include_router(admin_router, prefix=settings.API_V1_STR)
    app.include_router(documents_router, prefix=settings.API_V1_STR)
    app.include_router(chat_router, prefix=settings.API_V1_STR)
    app.include_router(doctors_router, prefix=settings.API_V1_STR)

    @app.get("/", tags=["Root"])
    async def root() -> dict[str, str]:
        """Root endpoint returning a simple API welcome message."""
        return {
            "message": f"Welcome to {settings.PROJECT_NAME}",
            "version": settings.VERSION,
            "docs": "/docs",
            "health": f"{settings.API_V1_STR}/health"
        }

    return app


app = create_application()
