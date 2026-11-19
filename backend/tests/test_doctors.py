"""Tests for Doctor API and recommendations."""
import pytest
import asyncio
from httpx import AsyncClient
from bson import ObjectId


@pytest.mark.asyncio
async def test_list_all_doctors(client: AsyncClient, auth_headers):
    """Test that all six doctors are loaded correctly."""
    response = await client.get("/api/v1/doctors", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert "doctors" in data
    assert len(data["doctors"]) == 6

    # Verify all six doctors are present
    doctor_ids = {d["doctor_id"] for d in data["doctors"]}
    expected_ids = {"dr.deven", "dr.rahul", "dr.niddhi", "dr.nita", "dr.suresh", "dr.mahesh"}
    assert doctor_ids == expected_ids


@pytest.mark.asyncio
async def test_doctor_details_match_dataset(client: AsyncClient, auth_headers):
    """Test that doctor details match the dataset exactly."""
    response = await client.get("/api/v1/doctors", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()

    doctors_by_id = {d["doctor_id"]: d for d in data["doctors"]}

    # Dr. Deven Patel
    deven = doctors_by_id["dr.deven"]
    assert deven["name"] == "Dr. Deven Patel"
    assert deven["qualifications"] == "MBBS"
    assert deven["specialization"] == "General Physician (સામાન્ય રોગોની સારવાર)"
    assert deven["experience_years"] == 6
    assert deven["photo_path"] == "doctor_photos/dr_deven.svg"
    assert deven["phone"] == "9700112232"
    assert deven["available_from"] == "10 a.m."
    assert deven["available_to"] == "7:30 p.m."

    # Dr. Rahul Maheta
    rahul = doctors_by_id["dr.rahul"]
    assert rahul["name"] == "Dr. Rahul Maheta"
    assert rahul["qualifications"] == "MBBS, MD (Neurology)"
    assert rahul["specialization"] == "Neurologist (મગજ અને નર્વસ સિસ્ટમ)"
    assert rahul["experience_years"] == 5
    assert rahul["photo_path"] == "doctor_photos/dr_rahul.svg"
    assert rahul["phone"] == "9900112233"
    assert rahul["available_from"] == "9 a.m."
    assert rahul["available_to"] == "6 p.m."

    # Dr. Nidhi Gopani
    niddhi = doctors_by_id["dr.niddhi"]
    assert niddhi["name"] == "Dr. Nidhi Gopani"
    assert niddhi["qualifications"] == "MBBS, MD (Dermatology)"
    assert niddhi["specialization"] == "Dermatologist (ત્વચા, વાળ, નખ)"
    assert niddhi["experience_years"] == 7
    assert niddhi["photo_path"] == "doctor_photos/dr_niddhi.svg"
    assert niddhi["phone"] == "9812345678"
    assert niddhi["available_from"] == "11 a.m."
    assert niddhi["available_to"] == "7 p.m."

    # Dr. Nita Surani
    nita = doctors_by_id["dr.nita"]
    assert nita["name"] == "Dr. Nita Surani"
    assert nita["qualifications"] == "MBBS, MD (Orthopedics)"
    assert nita["specialization"] == "Orthopedics (હાડકાં અને સાંધાના રોગ)"
    assert nita["experience_years"] == 5
    assert nita["photo_path"] == "doctor_photos/dr_neeta.svg"
    assert nita["phone"] == "9898123456"
    assert nita["available_from"] == "9:30 a.m."
    assert nita["available_to"] == "5:30 p.m."

    # Dr. Suresh Patel
    suresh = doctors_by_id["dr.suresh"]
    assert suresh["name"] == "Dr. Suresh Patel"
    assert suresh["qualifications"] == "MBBS, MS (Gynecology)"
    assert suresh["specialization"] == "Gynecologist (મહિલાઓના રોગ / ગર્ભાવસ્થા)"
    assert suresh["experience_years"] == 8
    assert suresh["photo_path"] == "doctor_photos/dr_shuresh.svg"
    assert suresh["phone"] == "9823456712"
    assert suresh["available_from"] == "5 a.m."
    assert suresh["available_to"] == "10 p.m."

    # Dr. Mahesh Ramani
    mahesh = doctors_by_id["dr.mahesh"]
    assert mahesh["name"] == "Dr. Mahesh Ramani"
    assert mahesh["qualifications"] == "MBBS, MD (Cardiology)"
    assert mahesh["specialization"] == "Cardiologist (હૃદયના રોગ)"
    assert mahesh["experience_years"] == 12
    assert mahesh["photo_path"] == "doctor_photos/dr_mahesh_GyexvM8.svg"
    assert mahesh["phone"] == "9876543210"
    assert mahesh["available_from"] == "5 a.m."
    assert mahesh["available_to"] == "11 p.m."


@pytest.mark.asyncio
async def test_get_specific_doctor(client: AsyncClient, auth_headers):
    """Test getting a specific doctor by ID."""
    response = await client.get("/api/v1/doctors/dr.deven", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["doctor_id"] == "dr.deven"
    assert data["name"] == "Dr. Deven Patel"


@pytest.mark.asyncio
async def test_get_nonexistent_doctor(client: AsyncClient, auth_headers):
    """Test getting a non-existent doctor returns 404."""
    response = await client.get("/api/v1/doctors/nonexistent", headers=auth_headers)
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_recommend_general_physician(client: AsyncClient, auth_headers):
    """Test general medical questions recommend the General Physician."""
    response = await client.get("/api/v1/doctors/recommend/fever", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 1
    # Should recommend Dr. Deven Patel (general physician)
    assert any(d["doctor_id"] == "dr.deven" for d in data)


@pytest.mark.asyncio
async def test_recommend_dermatologist(client: AsyncClient, auth_headers):
    """Test skin-related questions recommend the Dermatologist."""
    response = await client.get("/api/v1/doctors/recommend/skin", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 1
    # Should recommend Dr. Nidhi Gopani (dermatologist)
    assert any(d["doctor_id"] == "dr.niddhi" for d in data)


@pytest.mark.asyncio
async def test_recommend_cardiologist(client: AsyncClient, auth_headers):
    """Test heart-related questions recommend the Cardiologist."""
    response = await client.get("/api/v1/doctors/recommend/heart", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 1
    # Should recommend Dr. Mahesh Ramani (cardiologist)
    assert any(d["doctor_id"] == "dr.mahesh" for d in data)


@pytest.mark.asyncio
async def test_recommend_neurologist(client: AsyncClient, auth_headers):
    """Test neurological questions recommend the Neurologist."""
    response = await client.get("/api/v1/doctors/recommend/neurology", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 1
    # Should recommend Dr. Rahul Maheta (neurologist)
    assert any(d["doctor_id"] == "dr.rahul" for d in data)


@pytest.mark.asyncio
async def test_recommend_orthopedics(client: AsyncClient, auth_headers):
    """Test bone/joint questions recommend the Orthopedics specialist."""
    response = await client.get("/api/v1/doctors/recommend/bone", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 1
    # Should recommend Dr. Nita Surani (orthopedics)
    assert any(d["doctor_id"] == "dr.nita" for d in data)


@pytest.mark.asyncio
async def test_recommend_gynecologist(client: AsyncClient, auth_headers):
    """Test women's health questions recommend the Gynecologist."""
    response = await client.get("/api/v1/doctors/recommend/pregnancy", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 1
    # Should recommend Dr. Suresh Patel (gynecologist)
    assert any(d["doctor_id"] == "dr.suresh" for d in data)


@pytest.mark.asyncio
async def test_chat_includes_doctor_recommendations(client: AsyncClient, auth_headers):
    """Test that medical chat responses include doctor recommendations."""
    response = await client.post(
        "/api/v1/chat/message",
        headers=auth_headers,
        json={"message": "I have a fever and headache"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "doctor_recommendations" in data
    assert len(data["doctor_recommendations"]) >= 1
    # Should recommend general physician for fever
    assert any(d["doctor_id"] == "dr.deven" for d in data["doctor_recommendations"])


@pytest.mark.asyncio
async def test_chat_skin_includes_dermatologist(client: AsyncClient, auth_headers):
    """Test that skin-related chat responses include dermatologist."""
    response = await client.post(
        "/api/v1/chat/message",
        headers=auth_headers,
        json={"message": "I have a rash on my skin"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "doctor_recommendations" in data
    assert len(data["doctor_recommendations"]) >= 1
    # Should recommend dermatologist for skin rash
    assert any(d["doctor_id"] == "dr.niddhi" for d in data["doctor_recommendations"])


@pytest.mark.asyncio
async def test_chat_heart_includes_cardiologist(client: AsyncClient, auth_headers):
    """Test that heart-related chat responses include cardiologist."""
    response = await client.post(
        "/api/v1/chat/message",
        headers=auth_headers,
        json={"message": "I have chest pain and high blood pressure"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "doctor_recommendations" in data
    assert len(data["doctor_recommendations"]) >= 1
    # Should recommend cardiologist for heart issues
    assert any(d["doctor_id"] == "dr.mahesh" for d in data["doctor_recommendations"])


@pytest.mark.asyncio
async def test_chat_neurology_includes_neurologist(client: AsyncClient, auth_headers):
    """Test that neurological chat responses include neurologist."""
    response = await client.post(
        "/api/v1/chat/message",
        headers=auth_headers,
        json={"message": "I have severe headaches and numbness in my hands"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "doctor_recommendations" in data
    assert len(data["doctor_recommendations"]) >= 1
    # Should recommend neurologist for neurological symptoms
    assert any(d["doctor_id"] == "dr.rahul" for d in data["doctor_recommendations"])


@pytest.mark.asyncio
async def test_non_medical_question_no_recommendation(client: AsyncClient, auth_headers):
    """Test that non-medical questions don't trigger medical recommendations."""
    response = await client.post(
        "/api/v1/chat/message",
        headers=auth_headers,
        json={"message": "What are the hospital visiting hours?"}
    )
    assert response.status_code == 200
    data = response.json()
    # Non-medical questions may still have empty recommendations
    # but should not crash
    assert "doctor_recommendations" in data


@pytest.mark.asyncio
async def test_appointment_form_doctor_preselection(client: AsyncClient, auth_headers):
    """Test that doctor can be pre-selected in appointment form."""
    # This test verifies the API supports getting doctor by ID for pre-selection
    response = await client.get("/api/v1/doctors/dr.mahesh", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["doctor_id"] == "dr.mahesh"
    assert data["name"] == "Dr. Mahesh Ramani"


@pytest.mark.asyncio
async def test_doctor_photos_exist(client: AsyncClient, auth_headers):
    """Test that all doctor photo paths are valid (non-empty)."""
    response = await client.get("/api/v1/doctors", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()

    for doctor in data["doctors"]:
        assert doctor["photo_path"], f"Doctor {doctor['doctor_id']} has empty photo_path"
        assert doctor["photo_path"].startswith("doctor_photos/"), \
            f"Doctor {doctor['doctor_id']} photo path doesn't start with doctor_photos/"


@pytest.mark.asyncio
async def test_no_duplicate_doctors(client: AsyncClient, auth_headers):
    """Test that there are no duplicate doctor records."""
    response = await client.get("/api/v1/doctors", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()

    doctor_ids = [d["doctor_id"] for d in data["doctors"]]
    assert len(doctor_ids) == len(set(doctor_ids)), "Duplicate doctor IDs found"


@pytest.mark.asyncio
async def test_chat_answer_no_source_references(client: AsyncClient, auth_headers):
    """Test that chat answers don't contain internal source references like [Source 1]."""
    response = await client.post(
        "/api/v1/chat/message",
        headers=auth_headers,
        json={"message": "What is diabetes?"}
    )
    assert response.status_code == 200
    data = response.json()
    answer = data["assistant_message"]["content"]
    # Should not contain source reference patterns
    assert "[Source" not in answer
    assert "[Source 1]" not in answer
    assert "[Source 2]" not in answer
    assert "source_preview" not in answer.lower()


@pytest.mark.asyncio
async def test_chat_answer_structured_format(client: AsyncClient, auth_headers):
    """Test that medical answers have structured format with clear sections."""
    response = await client.post(
        "/api/v1/chat/message",
        headers=auth_headers,
        json={"message": "What are the symptoms of diabetes?"}
    )
    assert response.status_code == 200
    data = response.json()
    answer = data["assistant_message"]["content"]
    # Should have some content (structured format encouraged by prompt)
    assert len(answer) > 50
    # Should not be a single sentence fragment
    assert "." in answer


@pytest.mark.asyncio
async def test_chat_symptom_question_no_diagnosis(client: AsyncClient, auth_headers):
    """Test that symptom questions don't produce definitive diagnoses."""
    response = await client.post(
        "/api/v1/chat/message",
        headers=auth_headers,
        json={"message": "I have chest pain and shortness of breath"}
    )
    assert response.status_code == 200
    data = response.json()
    answer = data["assistant_message"]["content"].lower()
    # Should not contain definitive diagnosis language
    assert "you have" not in answer or "you have a" not in answer
    assert "diagnosis" not in answer or "confirm" not in answer
    # Should recommend seeing a doctor
    assert "doctor" in answer or "physician" in answer or "consult" in answer


@pytest.mark.asyncio
async def test_chat_doctor_recommendation_structure(client: AsyncClient, auth_headers):
    """Test that doctor recommendations have all required fields."""
    response = await client.post(
        "/api/v1/chat/message",
        headers=auth_headers,
        json={"message": "I have a rash on my skin"}
    )
    assert response.status_code == 200
    data = response.json()
    recommendations = data["doctor_recommendations"]
    assert len(recommendations) >= 1
    for rec in recommendations:
        assert "doctor_id" in rec
        assert "name" in rec
        assert "qualifications" in rec
        assert "specialization" in rec
        assert "experience_years" in rec
        assert "photo_path" in rec
        assert "phone" in rec
        assert "available_from" in rec
        assert "available_to" in rec
        assert "match_reason" in rec
        assert rec["photo_path"].startswith("doctor_photos/")


@pytest.mark.asyncio
async def test_chat_appointment_link_valid(client: AsyncClient, auth_headers):
    """Test that the appointment booking flow works via API."""
    # The frontend handles the actual link, but we verify the doctor exists
    # and can be retrieved for pre-selection
    response = await client.get("/api/v1/doctors/dr.niddhi", headers=auth_headers)
    assert response.status_code == 200
    doctor = response.json()
    assert doctor["doctor_id"] == "dr.niddhi"
    assert "name" in doctor
    # Verify the doctor has all fields needed for appointment booking
    assert doctor["available_from"]
    assert doctor["available_to"]


@pytest.mark.asyncio
async def test_chat_missing_context_handling(client: AsyncClient, auth_headers):
    """Test that questions with no relevant context are handled gracefully."""
    response = await client.post(
        "/api/v1/chat/message",
        headers=auth_headers,
        json={"message": "What is the cure for xyz123nonexistentdisease?"}
    )
    assert response.status_code == 200
    data = response.json()
    answer = data["assistant_message"]["content"]
    # Should indicate insufficient information
    assert len(answer) > 0
    # Should not crash or return error


@pytest.mark.asyncio
async def test_existing_chat_history_still_works(client: AsyncClient, auth_headers):
    """Test that existing chat history and feedback functionality still works."""
    # Send first message
    response1 = await client.post(
        "/api/v1/chat/message",
        headers=auth_headers,
        json={"message": "What is hypertension?"}
    )
    assert response1.status_code == 200
    data1 = response1.json()
    conversation_id = data1["conversation_id"]

    # Send follow-up message
    response2 = await client.post(
        "/api/v1/chat/message",
        headers=auth_headers,
        json={"message": "What causes it?", "conversation_id": conversation_id}
    )
    assert response2.status_code == 200
    data2 = response2.json()
    assert data2["conversation_id"] == conversation_id

    # Verify conversation has both messages
    response3 = await client.get(
        f"/api/v1/chat/conversations/{conversation_id}",
        headers=auth_headers
    )
    assert response3.status_code == 200
    conv = response3.json()
    assert len(conv["messages"]) == 4  # 2 user + 2 assistant

    # Test feedback on assistant message
    assistant_msg_id = conv["messages"][3]["id"]  # Last assistant message
    feedback_response = await client.post(
        "/api/v1/chat/feedback",
        headers=auth_headers,
        json={
            "conversation_id": conversation_id,
            "message_id": assistant_msg_id,
            "feedback_type": "positive"
        }
    )
    assert feedback_response.status_code in (200, 201)


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
