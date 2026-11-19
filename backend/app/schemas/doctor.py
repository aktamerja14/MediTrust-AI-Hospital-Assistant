"""Doctor request/response schemas."""
import re
from typing import Annotated
from pydantic import BaseModel, Field, ConfigDict, field_validator, StringConstraints


def validate_not_empty_or_whitespace(v: str) -> str:
    """Validate that string is not empty or whitespace-only."""
    if isinstance(v, str):
        stripped = v.strip()
        if not stripped:
            raise ValueError("Field cannot be empty or whitespace-only")
        return stripped
    return v


def validate_phone(v: str) -> str:
    """Validate phone number with international format support."""
    if not isinstance(v, str):
        raise ValueError("Phone must be a string")

    stripped = v.strip()
    if not stripped:
        raise ValueError("Phone cannot be empty or whitespace-only")

    allowed_chars = set("0123456789+ -()")
    if any(ch not in allowed_chars for ch in stripped):
        raise ValueError("Phone contains invalid characters (only digits, +, spaces, hyphens, parentheses allowed)")

    digits = "".join(ch for ch in stripped if ch.isdigit())
    if len(digits) < 10:
        raise ValueError("Phone number must contain at least 10 digits")
    if len(digits) > 15:
        raise ValueError("Phone number cannot exceed 15 digits")

    return stripped


def validate_photo_path(v: str) -> str:
    """Validate photo_path is a safe relative path under doctor_photos/."""
    if not isinstance(v, str):
        raise ValueError("photo_path must be a string")

    stripped = v.strip()
    if not stripped:
        raise ValueError("photo_path cannot be empty or whitespace-only")

    if stripped.startswith("/") or stripped.startswith("\\") or re.match(r"^[A-Za-z]:[\\/]", stripped):
        raise ValueError("photo_path must be a relative path (no absolute paths)")

    if re.search(r"[\\/]\.\.[\\/]", stripped) or stripped.startswith("..") or stripped.endswith(".."):
        raise ValueError("photo_path must not contain directory traversal")

    if ".." in stripped:
        raise ValueError("photo_path must not contain directory traversal")

    if not stripped.startswith("doctor_photos/"):
        raise ValueError("photo_path must be under doctor_photos/ directory")

    allowed_extensions = (".svg", ".jpg", ".jpeg", ".png")
    if not stripped.lower().endswith(allowed_extensions):
        raise ValueError(f"photo_path must have a valid image extension: {allowed_extensions}")

    return stripped


def validate_time_format(v: str) -> str:
    """Validate time format like '10 a.m.', '9:30 a.m.', '5 p.m.'."""
    if not isinstance(v, str):
        raise ValueError("Time must be a string")

    stripped = v.strip()
    if not stripped:
        raise ValueError("Time cannot be empty or whitespace-only")

    pattern = r"^(\d{1,2})(:\d{2})?\s+(a\.m\.|p\.m\.)$"
    match = re.match(pattern, stripped, re.IGNORECASE)
    if not match:
        raise ValueError("Time must be in format like '10 a.m.', '9:30 a.m.', '5 p.m.'")

    hour = int(match.group(1))
    if hour < 1 or hour > 12:
        raise ValueError("Hour must be between 1 and 12 for 12-hour format")

    minute = match.group(2)
    if minute:
        minute_val = int(minute[1:])
        if minute_val >= 60:
            raise ValueError("Minutes must be between 0 and 59")

    return stripped


DoctorId = Annotated[str, StringConstraints(min_length=1, max_length=64)]
DoctorName = Annotated[str, StringConstraints(min_length=1, max_length=128)]
Qualifications = Annotated[str, StringConstraints(min_length=1, max_length=256)]
Specialization = Annotated[str, StringConstraints(min_length=1, max_length=256)]
PhotoPath = Annotated[str, StringConstraints(min_length=1, max_length=256)]
Phone = Annotated[str, StringConstraints(min_length=1, max_length=32)]
TimeStr = Annotated[str, StringConstraints(min_length=1, max_length=16)]
MatchReason = Annotated[str, StringConstraints(min_length=1, max_length=512)]


class DoctorBase(BaseModel):
    """Base doctor schema with common fields."""
    doctor_id: DoctorId = Field(..., description="Unique doctor identifier (username)")
    name: DoctorName = Field(..., description="Doctor's full name")
    qualifications: Qualifications = Field(..., description="Doctor's qualifications")
    specialization: Specialization = Field(..., description="Doctor's specialization")
    experience_years: int = Field(..., ge=0, le=100, description="Years of experience")
    photo_path: PhotoPath = Field(..., description="Path to doctor's photo")
    phone: Phone = Field(..., description="Doctor's phone number")
    available_from: TimeStr = Field(..., description="Consultation start time")
    available_to: TimeStr = Field(..., description="Consultation end time")

    _strip_and_check_not_empty = field_validator(
        "doctor_id", "name", "qualifications", "specialization",
        mode="before"
    )(validate_not_empty_or_whitespace)

    _validate_phone = field_validator("phone", mode="before")(validate_phone)
    _validate_photo_path = field_validator("photo_path", mode="before")(validate_photo_path)
    _validate_time_format = field_validator("available_from", "available_to", mode="before")(validate_time_format)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "doctor_id": "dr.deven",
                "name": "Dr. Deven Patel",
                "qualifications": "MBBS",
                "specialization": "General Physician (સામાન્ય રોગોની સારવાર)",
                "experience_years": 6,
                "photo_path": "doctor_photos/dr_deven.svg",
                "phone": "9700112232",
                "available_from": "10 a.m.",
                "available_to": "7:30 p.m."
            }
        }
    )


class DoctorResponse(DoctorBase):
    """Doctor response schema."""
    id: str = Field(..., description="MongoDB document ID")

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": "507f1f77bcf86cd799439011",
                "doctor_id": "dr.deven",
                "name": "Dr. Deven Patel",
                "qualifications": "MBBS",
                "specialization": "General Physician (સામાન્ય રોગોની સારવાર)",
                "experience_years": 6,
                "photo_path": "doctor_photos/dr_deven.svg",
                "phone": "9700112232",
                "available_from": "10 a.m.",
                "available_to": "7:30 p.m."
            }
        }
    )


class DoctorListResponse(BaseModel):
    """Response for listing doctors."""
    doctors: list[DoctorResponse]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "doctors": [
                    {
                        "id": "507f1f77bcf86cd799439011",
                        "doctor_id": "dr.deven",
                        "name": "Dr. Deven Patel",
                        "qualifications": "MBBS",
                        "specialization": "General Physician (સામાન્ય રોગોની સારવાર)",
                        "experience_years": 6,
                        "photo_path": "doctor_photos/dr_deven.svg",
                        "phone": "9700112232",
                        "available_from": "10 a.m.",
                        "available_to": "7:30 p.m."
                    }
                ]
            }
        }
    )


class DoctorRecommendation(DoctorBase):
    """Doctor recommendation for chat responses."""
    match_reason: MatchReason = Field(..., description="Why this doctor was recommended")

    _strip_match_reason = field_validator("match_reason", mode="before")(validate_not_empty_or_whitespace)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "doctor_id": "dr.deven",
                "name": "Dr. Deven Patel",
                "qualifications": "MBBS",
                "specialization": "General Physician (સામાન્ય રોગોની સારવાર)",
                "experience_years": 6,
                "photo_path": "doctor_photos/dr_deven.svg",
                "phone": "9700112232",
                "available_from": "10 a.m.",
                "available_to": "7:30 p.m.",
                "match_reason": "General physician for common illnesses and fever"
            }
        }
    )
