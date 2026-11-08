"""Admin API routes."""
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, status
from pymongo.asynchronous.database import AsyncDatabase

from app.database.connection import get_db
from app.schemas.admin import AdminDashboardStats
from app.services.auth import AuthService
from app.api.dependencies import get_auth_service, require_admin

logger = logging.getLogger("meditrust.api.admin")

router = APIRouter(prefix="/admin", tags=["Admin"])


@router.get(
    "/dashboard",
    response_model=AdminDashboardStats,
    summary="Get admin dashboard statistics",
    responses={
        200: {"description": "Dashboard statistics retrieved successfully"},
        401: {"description": "Not authenticated"},
        403: {"description": "Insufficient permissions"},
    },
)
async def get_dashboard(
    auth_service: Annotated[AuthService, Depends(get_auth_service)],
    _admin: Annotated[None, Depends(require_admin)],
) -> AdminDashboardStats:
    """
    Get dashboard statistics for admin users.

    Returns:
        - Total registered users
        - Total patients
        - Total admins
        - Active users
        - Inactive users

    Requires valid admin JWT access token in Authorization header.
    """
    stats = await auth_service.get_dashboard_stats()
    logger.info("Admin dashboard accessed by admin user")
    return AdminDashboardStats(**stats)