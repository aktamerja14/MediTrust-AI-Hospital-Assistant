"""Tests for Doctor schema validation."""
import pytest
from pydantic import ValidationError

from app.schemas.doctor import DoctorBase, DoctorRecommendation


class TestDoctorBaseValidation:
    """Tests for DoctorBase schema validation."""

    def test_valid_doctor(self):
        """Test that a valid doctor passes validation."""
        data = {
            "doctor_id": "dr.test",
            "name": "Dr. Test",
            "qualifications": "MBBS",
            "specialization": "General Physician",
            "experience_years": 5,
            "photo_path": "doctor_photos/dr_test.svg",
            "phone": "9876543210",
            "available_from": "10 a.m.",
            "available_to": "6 p.m.",
        }
        doctor = DoctorBase(**data)
        assert doctor.doctor_id == "dr.test"
        assert doctor.name == "Dr. Test"

    def test_rejects_empty_doctor_id(self):
        """Test that empty doctor_id is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            DoctorBase(
                doctor_id="",
                name="Dr. Test",
                qualifications="MBBS",
                specialization="General",
                experience_years=5,
                photo_path="doctor_photos/dr_test.svg",
                phone="9876543210",
                available_from="10 a.m.",
                available_to="6 p.m.",
            )
        assert "empty or whitespace-only" in str(exc_info.value)

    def test_rejects_whitespace_only_doctor_id(self):
        """Test that whitespace-only doctor_id is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            DoctorBase(
                doctor_id="   ",
                name="Dr. Test",
                qualifications="MBBS",
                specialization="General",
                experience_years=5,
                photo_path="doctor_photos/dr_test.svg",
                phone="9876543210",
                available_from="10 a.m.",
                available_to="6 p.m.",
            )
        assert "empty or whitespace-only" in str(exc_info.value)

    def test_rejects_whitespace_only_name(self):
        """Test that whitespace-only name is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            DoctorBase(
                doctor_id="dr.test",
                name="   ",
                qualifications="MBBS",
                specialization="General",
                experience_years=5,
                photo_path="doctor_photos/dr_test.svg",
                phone="9876543210",
                available_from="10 a.m.",
                available_to="6 p.m.",
            )
        assert "empty or whitespace-only" in str(exc_info.value)

    def test_rejects_whitespace_only_qualifications(self):
        """Test that whitespace-only qualifications is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            DoctorBase(
                doctor_id="dr.test",
                name="Dr. Test",
                qualifications="   ",
                specialization="General",
                experience_years=5,
                photo_path="doctor_photos/dr_test.svg",
                phone="9876543210",
                available_from="10 a.m.",
                available_to="6 p.m.",
            )
        assert "empty or whitespace-only" in str(exc_info.value)

    def test_rejects_whitespace_only_specialization(self):
        """Test that whitespace-only specialization is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            DoctorBase(
                doctor_id="dr.test",
                name="Dr. Test",
                qualifications="MBBS",
                specialization="   ",
                experience_years=5,
                photo_path="doctor_photos/dr_test.svg",
                phone="9876543210",
                available_from="10 a.m.",
                available_to="6 p.m.",
            )
        assert "empty or whitespace-only" in str(exc_info.value)

    def test_rejects_negative_experience(self):
        """Test that negative experience_years is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            DoctorBase(
                doctor_id="dr.test",
                name="Dr. Test",
                qualifications="MBBS",
                specialization="General",
                experience_years=-1,
                photo_path="doctor_photos/dr_test.svg",
                phone="9876543210",
                available_from="10 a.m.",
                available_to="6 p.m.",
            )
        assert "greater than or equal to 0" in str(exc_info.value)

    def test_rejects_excessive_experience(self):
        """Test that experience_years > 100 is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            DoctorBase(
                doctor_id="dr.test",
                name="Dr. Test",
                qualifications="MBBS",
                specialization="General",
                experience_years=101,
                photo_path="doctor_photos/dr_test.svg",
                phone="9876543210",
                available_from="10 a.m.",
                available_to="6 p.m.",
            )
        assert "less than or equal to 100" in str(exc_info.value)

    def test_rejects_phone_with_less_than_10_digits(self):
        """Test that phone with < 10 digits is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            DoctorBase(
                doctor_id="dr.test",
                name="Dr. Test",
                qualifications="MBBS",
                specialization="General",
                experience_years=5,
                photo_path="doctor_photos/dr_test.svg",
                phone="12345",
                available_from="10 a.m.",
                available_to="6 p.m.",
            )
        assert "at least 10 digits" in str(exc_info.value)

    def test_accepts_international_phone_format(self):
        """Test that international phone format with 10+ digits is accepted."""
        data = {
            "doctor_id": "dr.test",
            "name": "Dr. Test",
            "qualifications": "MBBS",
            "specialization": "General",
            "experience_years": 5,
            "photo_path": "doctor_photos/dr_test.svg",
            "phone": "+91 98765 43210",
            "available_from": "10 a.m.",
            "available_to": "6 p.m.",
        }
        doctor = DoctorBase(**data)
        assert doctor.phone == "+91 98765 43210"

    def test_accepts_phone_with_hyphens(self):
        """Test that phone with hyphens is accepted."""
        data = {
            "doctor_id": "dr.test",
            "name": "Dr. Test",
            "qualifications": "MBBS",
            "specialization": "General",
            "experience_years": 5,
            "photo_path": "doctor_photos/dr_test.svg",
            "phone": "987-654-3210",
            "available_from": "10 a.m.",
            "available_to": "6 p.m.",
        }
        doctor = DoctorBase(**data)
        assert doctor.phone == "987-654-3210"

    def test_accepts_phone_with_parentheses(self):
        """Test that phone with parentheses is accepted."""
        data = {
            "doctor_id": "dr.test",
            "name": "Dr. Test",
            "qualifications": "MBBS",
            "specialization": "General",
            "experience_years": 5,
            "photo_path": "doctor_photos/dr_test.svg",
            "phone": "(987) 654-3210",
            "available_from": "10 a.m.",
            "available_to": "6 p.m.",
        }
        doctor = DoctorBase(**data)
        assert doctor.phone == "(987) 654-3210"

    def test_accepts_phone_with_spaces(self):
        """Test that phone with spaces is accepted."""
        data = {
            "doctor_id": "dr.test",
            "name": "Dr. Test",
            "qualifications": "MBBS",
            "specialization": "General",
            "experience_years": 5,
            "photo_path": "doctor_photos/dr_test.svg",
            "phone": "987 654 3210",
            "available_from": "10 a.m.",
            "available_to": "6 p.m.",
        }
        doctor = DoctorBase(**data)
        assert doctor.phone == "987 654 3210"

    def test_rejects_phone_with_alphabetic_characters(self):
        """Test that phone with alphabetic characters is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            DoctorBase(
                doctor_id="dr.test",
                name="Dr. Test",
                qualifications="MBBS",
                specialization="General",
                experience_years=5,
                photo_path="doctor_photos/dr_test.svg",
                phone="987-abc-3210",
                available_from="10 a.m.",
                available_to="6 p.m.",
            )
        assert "invalid characters" in str(exc_info.value)

    def test_rejects_phone_with_invalid_symbols(self):
        """Test that phone with invalid symbols is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            DoctorBase(
                doctor_id="dr.test",
                name="Dr. Test",
                qualifications="MBBS",
                specialization="General",
                experience_years=5,
                photo_path="doctor_photos/dr_test.svg",
                phone="987@654#3210",
                available_from="10 a.m.",
                available_to="6 p.m.",
            )
        assert "invalid characters" in str(exc_info.value)

    def test_rejects_phone_exceeding_15_digits(self):
        """Test that phone with > 15 digits is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            DoctorBase(
                doctor_id="dr.test",
                name="Dr. Test",
                qualifications="MBBS",
                specialization="General",
                experience_years=5,
                photo_path="doctor_photos/dr_test.svg",
                phone="+1 987654321012345",
                available_from="10 a.m.",
                available_to="6 p.m.",
            )
        assert "cannot exceed 15 digits" in str(exc_info.value)

    def test_rejects_whitespace_only_phone(self):
        """Test that whitespace-only phone is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            DoctorBase(
                doctor_id="dr.test",
                name="Dr. Test",
                qualifications="MBBS",
                specialization="General",
                experience_years=5,
                photo_path="doctor_photos/dr_test.svg",
                phone="   ",
                available_from="10 a.m.",
                available_to="6 p.m.",
            )
        assert "empty or whitespace-only" in str(exc_info.value)

    def test_rejects_absolute_photo_path_unix(self):
        """Test that Unix absolute photo_path is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            DoctorBase(
                doctor_id="dr.test",
                name="Dr. Test",
                qualifications="MBBS",
                specialization="General",
                experience_years=5,
                photo_path="/doctor_photos/dr_test.svg",
                phone="9876543210",
                available_from="10 a.m.",
                available_to="6 p.m.",
            )
        assert "relative path" in str(exc_info.value)

    def test_rejects_absolute_photo_path_windows(self):
        """Test that Windows absolute photo_path is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            DoctorBase(
                doctor_id="dr.test",
                name="Dr. Test",
                qualifications="MBBS",
                specialization="General",
                experience_years=5,
                photo_path="C:\\doctor_photos\\dr_test.svg",
                phone="9876543210",
                available_from="10 a.m.",
                available_to="6 p.m.",
            )
        assert "relative path" in str(exc_info.value)

    def test_rejects_traversal_photo_path_forward_slash(self):
        """Test that forward-slash directory traversal in photo_path is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            DoctorBase(
                doctor_id="dr.test",
                name="Dr. Test",
                qualifications="MBBS",
                specialization="General",
                experience_years=5,
                photo_path="doctor_photos/../dr_test.svg",
                phone="9876543210",
                available_from="10 a.m.",
                available_to="6 p.m.",
            )
        assert "traversal" in str(exc_info.value)

    def test_rejects_traversal_photo_path_backslash(self):
        """Test that backslash directory traversal in photo_path is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            DoctorBase(
                doctor_id="dr.test",
                name="Dr. Test",
                qualifications="MBBS",
                specialization="General",
                experience_years=5,
                photo_path="doctor_photos\\..\\dr_test.svg",
                phone="9876543210",
                available_from="10 a.m.",
                available_to="6 p.m.",
            )
        assert "traversal" in str(exc_info.value)

    def test_rejects_traversal_at_start(self):
        """Test that photo_path starting with .. is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            DoctorBase(
                doctor_id="dr.test",
                name="Dr. Test",
                qualifications="MBBS",
                specialization="General",
                experience_years=5,
                photo_path="../doctor_photos/dr_test.svg",
                phone="9876543210",
                available_from="10 a.m.",
                available_to="6 p.m.",
            )
        assert "traversal" in str(exc_info.value)

    def test_rejects_traversal_at_end(self):
        """Test that photo_path ending with .. is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            DoctorBase(
                doctor_id="dr.test",
                name="Dr. Test",
                qualifications="MBBS",
                specialization="General",
                experience_years=5,
                photo_path="doctor_photos/dr_test.svg/..",
                phone="9876543210",
                available_from="10 a.m.",
                available_to="6 p.m.",
            )
        assert "traversal" in str(exc_info.value)

    def test_rejects_wrong_directory_photo_path(self):
        """Test that photo_path outside doctor_photos/ is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            DoctorBase(
                doctor_id="dr.test",
                name="Dr. Test",
                qualifications="MBBS",
                specialization="General",
                experience_years=5,
                photo_path="other_photos/dr_test.svg",
                phone="9876543210",
                available_from="10 a.m.",
                available_to="6 p.m.",
            )
        assert "doctor_photos" in str(exc_info.value)

    def test_rejects_whitespace_only_photo_path(self):
        """Test that whitespace-only photo_path is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            DoctorBase(
                doctor_id="dr.test",
                name="Dr. Test",
                qualifications="MBBS",
                specialization="General",
                experience_years=5,
                photo_path="   ",
                phone="9876543210",
                available_from="10 a.m.",
                available_to="6 p.m.",
            )
        assert "empty or whitespace-only" in str(exc_info.value)

    def test_accepts_svg_extension(self):
        """Test that .svg extension is accepted."""
        data = {
            "doctor_id": "dr.test",
            "name": "Dr. Test",
            "qualifications": "MBBS",
            "specialization": "General",
            "experience_years": 5,
            "photo_path": "doctor_photos/dr_test.svg",
            "phone": "9876543210",
            "available_from": "10 a.m.",
            "available_to": "6 p.m.",
        }
        doctor = DoctorBase(**data)
        assert doctor.photo_path == "doctor_photos/dr_test.svg"

    def test_accepts_jpg_extension(self):
        """Test that .jpg extension is accepted (backward compat)."""
        data = {
            "doctor_id": "dr.test",
            "name": "Dr. Test",
            "qualifications": "MBBS",
            "specialization": "General",
            "experience_years": 5,
            "photo_path": "doctor_photos/dr_test.jpg",
            "phone": "9876543210",
            "available_from": "10 a.m.",
            "available_to": "6 p.m.",
        }
        doctor = DoctorBase(**data)
        assert doctor.photo_path == "doctor_photos/dr_test.jpg"

    def test_accepts_jpeg_extension(self):
        """Test that .jpeg extension is accepted."""
        data = {
            "doctor_id": "dr.test",
            "name": "Dr. Test",
            "qualifications": "MBBS",
            "specialization": "General",
            "experience_years": 5,
            "photo_path": "doctor_photos/dr_test.jpeg",
            "phone": "9876543210",
            "available_from": "10 a.m.",
            "available_to": "6 p.m.",
        }
        doctor = DoctorBase(**data)
        assert doctor.photo_path == "doctor_photos/dr_test.jpeg"

    def test_accepts_png_extension(self):
        """Test that .png extension is accepted."""
        data = {
            "doctor_id": "dr.test",
            "name": "Dr. Test",
            "qualifications": "MBBS",
            "specialization": "General",
            "experience_years": 5,
            "photo_path": "doctor_photos/dr_test.png",
            "phone": "9876543210",
            "available_from": "10 a.m.",
            "available_to": "6 p.m.",
        }
        doctor = DoctorBase(**data)
        assert doctor.photo_path == "doctor_photos/dr_test.png"

    def test_rejects_invalid_extension(self):
        """Test that invalid extension is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            DoctorBase(
                doctor_id="dr.test",
                name="Dr. Test",
                qualifications="MBBS",
                specialization="General",
                experience_years=5,
                photo_path="doctor_photos/dr_test.gif",
                phone="9876543210",
                available_from="10 a.m.",
                available_to="6 p.m.",
            )
        assert "valid image extension" in str(exc_info.value)

    def test_rejects_no_extension(self):
        """Test that photo_path without extension is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            DoctorBase(
                doctor_id="dr.test",
                name="Dr. Test",
                qualifications="MBBS",
                specialization="General",
                experience_years=5,
                photo_path="doctor_photos/dr_test",
                phone="9876543210",
                available_from="10 a.m.",
                available_to="6 p.m.",
            )
        assert "valid image extension" in str(exc_info.value)

    def test_accepts_valid_time_formats(self):
        """Test that valid time formats are accepted."""
        valid_times = [
            "10 a.m.", "9:30 a.m.", "5 p.m.", "11 p.m.", "12 a.m.", "12 p.m.",
            "1 a.m.", "1:05 p.m.", "9:00 a.m.", "5:30 p.m.", "10 a.m.", "7:30 p.m."
        ]
        for time_str in valid_times:
            data = {
                "doctor_id": "dr.test",
                "name": "Dr. Test",
                "qualifications": "MBBS",
                "specialization": "General",
                "experience_years": 5,
                "photo_path": "doctor_photos/dr_test.svg",
                "phone": "9876543210",
                "available_from": time_str,
                "available_to": "6 p.m.",
            }
            doctor = DoctorBase(**data)
            assert doctor.available_from == time_str.strip()

    def test_rejects_invalid_time_format(self):
        """Test that invalid time format is rejected."""
        invalid_times = [
            "10:00", "10 AM", "10:00 AM", "10am", "10.00 a.m.", "10a.m.",
            "10:00a.m.", "morning", "10:00:00 a.m.", "25 a.m.", "13 p.m.",
            "0 a.m.", "0 p.m.", "10:60 a.m.", "10:70 p.m."
        ]
        for time_str in invalid_times:
            with pytest.raises(ValidationError):
                DoctorBase(
                    doctor_id="dr.test",
                    name="Dr. Test",
                    qualifications="MBBS",
                    specialization="General",
                    experience_years=5,
                    photo_path="doctor_photos/dr_test.svg",
                    phone="9876543210",
                    available_from=time_str,
                    available_to="6 p.m.",
                )

    def test_rejects_whitespace_only_time(self):
        """Test that whitespace-only time is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            DoctorBase(
                doctor_id="dr.test",
                name="Dr. Test",
                qualifications="MBBS",
                specialization="General",
                experience_years=5,
                photo_path="doctor_photos/dr_test.svg",
                phone="9876543210",
                available_from="   ",
                available_to="6 p.m.",
            )
        assert "empty or whitespace-only" in str(exc_info.value)

    def test_existing_valid_doctor_records(self):
        """Test that all existing doctor records from database pass validation."""
        doctors = [
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
        for doc in doctors:
            doctor = DoctorBase(**doc)
            assert doctor.doctor_id == doc["doctor_id"]
            assert doctor.photo_path == doc["photo_path"]


class TestDoctorRecommendationValidation:
    """Tests for DoctorRecommendation schema validation."""

    def test_valid_recommendation(self):
        """Test that a valid recommendation passes validation."""
        data = {
            "doctor_id": "dr.test",
            "name": "Dr. Test",
            "qualifications": "MBBS",
            "specialization": "General Physician",
            "experience_years": 5,
            "photo_path": "doctor_photos/dr_test.svg",
            "phone": "9876543210",
            "available_from": "10 a.m.",
            "available_to": "6 p.m.",
            "match_reason": "General physician for common illnesses",
        }
        rec = DoctorRecommendation(**data)
        assert rec.match_reason == "General physician for common illnesses"

    def test_rejects_empty_match_reason(self):
        """Test that empty match_reason is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            DoctorRecommendation(
                doctor_id="dr.test",
                name="Dr. Test",
                qualifications="MBBS",
                specialization="General",
                experience_years=5,
                photo_path="doctor_photos/dr_test.svg",
                phone="9876543210",
                available_from="10 a.m.",
                available_to="6 p.m.",
                match_reason="",
            )
        assert "empty or whitespace-only" in str(exc_info.value)

    def test_rejects_whitespace_only_match_reason(self):
        """Test that whitespace-only match_reason is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            DoctorRecommendation(
                doctor_id="dr.test",
                name="Dr. Test",
                qualifications="MBBS",
                specialization="General",
                experience_years=5,
                photo_path="doctor_photos/dr_test.svg",
                phone="9876543210",
                available_from="10 a.m.",
                available_to="6 p.m.",
                match_reason="   ",
            )
        assert "empty or whitespace-only" in str(exc_info.value)

    def test_recommendation_applies_same_validators(self):
        """Test that DoctorRecommendation uses same validators as DoctorBase."""
        with pytest.raises(ValidationError) as exc_info:
            DoctorRecommendation(
                doctor_id="dr.test",
                name="Dr. Test",
                qualifications="MBBS",
                specialization="General",
                experience_years=5,
                photo_path="other_photos/dr_test.svg",
                phone="9876543210",
                available_from="10 a.m.",
                available_to="6 p.m.",
                match_reason="Test reason",
            )
        assert "doctor_photos" in str(exc_info.value)

    def test_recommendation_phone_validation(self):
        """Test that DoctorRecommendation phone validation works."""
        with pytest.raises(ValidationError) as exc_info:
            DoctorRecommendation(
                doctor_id="dr.test",
                name="Dr. Test",
                qualifications="MBBS",
                specialization="General",
                experience_years=5,
                photo_path="doctor_photos/dr_test.svg",
                phone="12345",
                available_from="10 a.m.",
                available_to="6 p.m.",
                match_reason="Test reason",
            )
        assert "at least 10 digits" in str(exc_info.value)

    def test_recommendation_accepts_international_phone(self):
        """Test that DoctorRecommendation accepts international phone."""
        data = {
            "doctor_id": "dr.test",
            "name": "Dr. Test",
            "qualifications": "MBBS",
            "specialization": "General",
            "experience_years": 5,
            "photo_path": "doctor_photos/dr_test.svg",
            "phone": "+91 98765 43210",
            "available_from": "10 a.m.",
            "available_to": "6 p.m.",
            "match_reason": "Test reason",
        }
        rec = DoctorRecommendation(**data)
        assert rec.phone == "+91 98765 43210"

    def test_recommendation_rejects_alphabetic_phone(self):
        """Test that DoctorRecommendation rejects alphabetic phone."""
        with pytest.raises(ValidationError) as exc_info:
            DoctorRecommendation(
                doctor_id="dr.test",
                name="Dr. Test",
                qualifications="MBBS",
                specialization="General",
                experience_years=5,
                photo_path="doctor_photos/dr_test.svg",
                phone="987-abc-3210",
                available_from="10 a.m.",
                available_to="6 p.m.",
                match_reason="Test reason",
            )
        assert "invalid characters" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])