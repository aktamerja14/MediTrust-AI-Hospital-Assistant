"""Admin API request/response schemas."""
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict


class AdminDashboardStats(BaseModel):
    """Admin dashboard statistics response."""
    total_users: int = Field(..., description="Total registered users")
    total_patients: int = Field(..., description="Total patients")
    total_admins: int = Field(..., description="Total admins")
    active_users: int = Field(..., description="Active users")
    inactive_users: int = Field(..., description="Inactive users")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "total_users": 100,
                "total_patients": 95,
                "total_admins": 5,
                "active_users": 98,
                "inactive_users": 2
            }
        }
    )