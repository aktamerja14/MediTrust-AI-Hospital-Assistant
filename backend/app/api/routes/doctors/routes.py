"""Doctor API routes."""
import logging
from typing import Annotated, Optional, List
from fastapi import APIRouter, Depends, HTTPException, status, Query
from pymongo.asynchronous.database import AsyncDatabase
from pymongo.errors import DuplicateKeyError
from bson import ObjectId

from app.database.connection import get_db
from app.api.dependencies import get_current_active_user, require_any_role
from app.schemas.auth import UserResponse
from app.schemas.doctor import DoctorResponse, DoctorListResponse, DoctorRecommendation

logger = logging.getLogger("meditrust.api.doctors")

router = APIRouter(prefix="/doctors", tags=["Doctors"])

DOCTORS_COLLECTION = "doctors"


def _doctor_doc_to_response(doc: dict) -> DoctorResponse:
    """Convert MongoDB document to DoctorResponse."""
    return DoctorResponse(
        id=str(doc["_id"]),
        doctor_id=doc["doctor_id"],
        name=doc["name"],
        qualifications=doc["qualifications"],
        specialization=doc["specialization"],
        experience_years=doc["experience_years"],
        photo_path=doc["photo_path"],
        phone=doc["phone"],
        available_from=doc["available_from"],
        available_to=doc["available_to"],
    )


# PUBLIC ENDPOINTS (no authentication required) - Must come before parameterized routes

@router.get(
    "/public",
    response_model=DoctorListResponse,
    summary="List all doctors (public access)",
    responses={
        200: {"description": "List of all doctors"},
    },
)
async def list_doctors_public(
    db: Annotated[AsyncDatabase, Depends(get_db)],
    specialization: Optional[str] = Query(None, description="Filter by specialization"),
) -> DoctorListResponse:
    """
    List all doctors with optional specialization filter.

    Public endpoint - no authentication required for website display.
    """
    query = {}
    if specialization:
        query["specialization"] = {"$regex": specialization, "$options": "i"}

    cursor = db[DOCTORS_COLLECTION].find(query).sort("name", 1)
    doctors = []
    async for doc in cursor:
        doctors.append(_doctor_doc_to_response(doc))

    return DoctorListResponse(doctors=doctors)


@router.get(
    "/public/{doctor_id}",
    response_model=DoctorResponse,
    summary="Get a specific doctor by ID (public access)",
    responses={
        200: {"description": "Doctor details"},
        404: {"description": "Doctor not found"},
    },
)
async def get_doctor_public(
    doctor_id: str,
    db: Annotated[AsyncDatabase, Depends(get_db)],
) -> DoctorResponse:
    """
    Get a specific doctor by their doctor_id (username).

    Public endpoint - no authentication required for website display.
    """
    doc = await db[DOCTORS_COLLECTION].find_one({"doctor_id": doctor_id})
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Doctor not found"
        )

    return _doctor_doc_to_response(doc)


# PROTECTED ENDPOINTS (authentication required)

@router.get(
    "",
    response_model=DoctorListResponse,
    summary="List all doctors",
    responses={
        200: {"description": "List of all doctors"},
        401: {"description": "Not authenticated"},
        403: {"description": "Inactive user account"},
    },
    dependencies=[Depends(require_any_role)],
)
async def list_doctors(
    current_user: Annotated[UserResponse, Depends(get_current_active_user)],
    db: Annotated[AsyncDatabase, Depends(get_db)],
    specialization: Optional[str] = Query(None, description="Filter by specialization"),
) -> DoctorListResponse:
    """
    List all doctors with optional specialization filter.

    Both patients and admins can use this endpoint.
    """
    query = {}
    if specialization:
        query["specialization"] = {"$regex": specialization, "$options": "i"}

    cursor = db[DOCTORS_COLLECTION].find(query).sort("name", 1)
    doctors = []
    async for doc in cursor:
        doctors.append(_doctor_doc_to_response(doc))

    return DoctorListResponse(doctors=doctors)


@router.get(
    "/{doctor_id}",
    response_model=DoctorResponse,
    summary="Get a specific doctor by ID",
    responses={
        200: {"description": "Doctor details"},
        401: {"description": "Not authenticated"},
        403: {"description": "Inactive user account"},
        404: {"description": "Doctor not found"},
    },
    dependencies=[Depends(require_any_role)],
)
async def get_doctor(
    doctor_id: str,
    current_user: Annotated[UserResponse, Depends(get_current_active_user)],
    db: Annotated[AsyncDatabase, Depends(get_db)],
) -> DoctorResponse:
    """
    Get a specific doctor by their doctor_id (username).

    Both patients and admins can use this endpoint.
    """
    doc = await db[DOCTORS_COLLECTION].find_one({"doctor_id": doctor_id})
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Doctor not found"
        )

    return _doctor_doc_to_response(doc)


@router.get(
    "/recommend/{specialization}",
    response_model=List[DoctorRecommendation],
    summary="Get doctor recommendations for a specialization",
    responses={
        200: {"description": "Recommended doctors"},
        401: {"description": "Not authenticated"},
        403: {"description": "Inactive user account"},
    },
    dependencies=[Depends(require_any_role)],
)
async def recommend_doctors(
    specialization: str,
    current_user: Annotated[UserResponse, Depends(get_current_active_user)],
    db: Annotated[AsyncDatabase, Depends(get_db)],
    limit: int = Query(3, ge=1, le=5, description="Maximum number of recommendations"),
) -> List[DoctorRecommendation]:
    """
    Get doctor recommendations based on medical specialization/concern.

    Maps user's medical concern to relevant specialists.
    """
    specialization_lower = specialization.lower()

    # Map specializations to doctor queries
    specialization_map = {
        "general": ["general physician", "general medicine"],
        "fever": ["general physician", "general medicine"],
        "cold": ["general physician", "general medicine"],
        "neurology": ["neurologist", "neurology"],
        "nerve": ["neurologist", "neurology"],
        "brain": ["neurologist", "neurology"],
        "skin": ["dermatologist", "dermatology"],
        "hair": ["dermatologist", "dermatology"],
        "nail": ["dermatologist", "dermatology"],
        "bone": ["orthopedics", "orthopedic"],
        "joint": ["orthopedics", "orthopedic"],
        "orthopedic": ["orthopedics", "orthopedic"],
        "women": ["gynecologist", "gynecology"],
        "pregnancy": ["gynecologist", "gynecology"],
        "gynecology": ["gynecologist", "gynecology"],
        "heart": ["cardiologist", "cardiology"],
        "cardiac": ["cardiologist", "cardiology"],
        "cardiology": ["cardiologist", "cardiology"],
    }

    # Find matching specialization
    search_terms = specialization_map.get(specialization_lower, [specialization_lower])

    # Build query for matching doctors
    or_conditions = []
    for term in search_terms:
        or_conditions.append({"specialization": {"$regex": term, "$options": "i"}})

    query = {"$or": or_conditions} if or_conditions else {}

    cursor = db[DOCTORS_COLLECTION].find(query).limit(limit)
    recommendations = []
    async for doc in cursor:
        # Determine match reason
        match_reason = f"Specialist in {doc['specialization']}"
        if "general" in specialization_lower or "fever" in specialization_lower or "cold" in specialization_lower:
            match_reason = "General physician for common illnesses and fever"
        elif "neurology" in specialization_lower or "nerve" in specialization_lower or "brain" in specialization_lower:
            match_reason = "Neurologist for neurological concerns and nerve-related symptoms"
        elif "skin" in specialization_lower or "hair" in specialization_lower or "nail" in specialization_lower:
            match_reason = "Dermatologist for skin, hair, and nail concerns"
        elif "bone" in specialization_lower or "joint" in specialization_lower or "orthopedic" in specialization_lower:
            match_reason = "Orthopedics specialist for bone and joint concerns"
        elif "women" in specialization_lower or "pregnancy" in specialization_lower or "gynecology" in specialization_lower:
            match_reason = "Gynecologist for women's health and pregnancy concerns"
        elif "heart" in specialization_lower or "cardiac" in specialization_lower or "cardiology" in specialization_lower:
            match_reason = "Cardiologist for heart-related concerns"

        recommendations.append(DoctorRecommendation(
            doctor_id=doc["doctor_id"],
            name=doc["name"],
            qualifications=doc["qualifications"],
            specialization=doc["specialization"],
            experience_years=doc["experience_years"],
            photo_path=doc["photo_path"],
            phone=doc["phone"],
            available_from=doc["available_from"],
            available_to=doc["available_to"],
            match_reason=match_reason,
        ))

    # If no specific match found, return general physician as fallback
    if not recommendations:
        general_doc = await db[DOCTORS_COLLECTION].find_one(
            {"specialization": {"$regex": "general physician", "$options": "i"}}
        )
        if general_doc:
            recommendations.append(DoctorRecommendation(
                doctor_id=general_doc["doctor_id"],
                name=general_doc["name"],
                qualifications=general_doc["qualifications"],
                specialization=general_doc["specialization"],
                experience_years=general_doc["experience_years"],
                photo_path=general_doc["photo_path"],
                phone=general_doc["phone"],
                available_from=general_doc["available_from"],
                available_to=general_doc["available_to"],
                match_reason="General physician for general medical questions",
            ))

    return recommendations


@router.post(
    "/populate",
    response_model=dict,
    summary="Populate doctors from dataset (admin only)",
    responses={
        200: {"description": "Doctors populated successfully"},
        401: {"description": "Not authenticated"},
        403: {"description": "Admin access required"},
    },
    dependencies=[Depends(require_any_role)],
)
async def populate_doctors(
    current_user: Annotated[UserResponse, Depends(get_current_active_user)],
    db: Annotated[AsyncDatabase, Depends(get_db)],
) -> dict:
    """
    Populate the doctors collection from the dataset.
    Only creates doctors that don't already exist (based on doctor_id).
    """
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )

    doctors_data = [
        {
            "doctor_id": "dr.deven",
            "name": "Dr. Deven Patel",
            "qualifications": "MBBS",
            "specialization": "General Physician (સામાન્ય રોગોની સારવાર)",
            "experience_years": 6,
            "photo_path": "doctor_photos/dr_deven.svg",
            "phone": "9700112232",
            "available_from": "10 a.m.",
            "available_to": "7:30 p.m.",
        },
        {
            "doctor_id": "dr.rahul",
            "name": "Dr. Rahul Maheta",
            "qualifications": "MBBS, MD (Neurology)",
            "specialization": "Neurologist (મગજ અને નર્વસ સિસ્ટમ)",
            "experience_years": 5,
            "photo_path": "doctor_photos/dr_rahul.svg",
            "phone": "9900112233",
            "available_from": "9 a.m.",
            "available_to": "6 p.m.",
        },
        {
            "doctor_id": "dr.niddhi",
            "name": "Dr. Nidhi Gopani",
            "qualifications": "MBBS, MD (Dermatology)",
            "specialization": "Dermatologist (ત્વચા, વાળ, નખ)",
            "experience_years": 7,
            "photo_path": "doctor_photos/dr_niddhi.svg",
            "phone": "9812345678",
            "available_from": "11 a.m.",
            "available_to": "7 p.m.",
        },
        {
            "doctor_id": "dr.nita",
            "name": "Dr. Nita Surani",
            "qualifications": "MBBS, MD (Orthopedics)",
            "specialization": "Orthopedics (હાડકાં અને સાંધાના રોગ)",
            "experience_years": 5,
            "photo_path": "doctor_photos/dr_neeta.svg",
            "phone": "9898123456",
            "available_from": "9:30 a.m.",
            "available_to": "5:30 p.m.",
        },
        {
            "doctor_id": "dr.suresh",
            "name": "Dr. Suresh Patel",
            "qualifications": "MBBS, MS (Gynecology)",
            "specialization": "Gynecologist (મહિલાઓના રોગ / ગર્ભાવસ્થા)",
            "experience_years": 8,
            "photo_path": "doctor_photos/dr_shuresh.svg",
            "phone": "9823456712",
            "available_from": "5 a.m.",
            "available_to": "10 p.m.",
        },
        {
            "doctor_id": "dr.mahesh",
            "name": "Dr. Mahesh Ramani",
            "qualifications": "MBBS, MD (Cardiology)",
            "specialization": "Cardiologist (હૃદયના રોગ)",
            "experience_years": 12,
            "photo_path": "doctor_photos/dr_mahesh_GyexvM8.svg",
            "phone": "9876543210",
            "available_from": "5 a.m.",
            "available_to": "11 p.m.",
        },
    ]

    created_count = 0
    skipped_count = 0

    for doctor in doctors_data:
        existing = await db[DOCTORS_COLLECTION].find_one({"doctor_id": doctor["doctor_id"]})
        if existing:
            skipped_count += 1
            continue

        try:
            await db[DOCTORS_COLLECTION].insert_one(doctor)
            created_count += 1
        except DuplicateKeyError:
            skipped_count += 1

    # Create index on doctor_id for uniqueness
    await db[DOCTORS_COLLECTION].create_index("doctor_id", unique=True)

    return {
        "message": "Doctor population complete",
        "created": created_count,
        "skipped": skipped_count,
        "total_in_dataset": len(doctors_data),
    }