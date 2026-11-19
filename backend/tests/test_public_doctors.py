"""Tests for Public Doctor API endpoints."""
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_list_all_doctors_public(client: AsyncClient):
    """Test that all six doctors are accessible via public endpoint without auth."""
    response = await client.get("/api/v1/doctors/public")
    assert response.status_code == 200
    data = response.json()
    assert "doctors" in data
    assert len(data["doctors"]) == 6

    # Verify all six doctors are present
    doctor_ids = {d["doctor_id"] for d in data["doctors"]}
    expected_ids = {"dr.deven", "dr.rahul", "dr.niddhi", "dr.nita", "dr.suresh", "dr.mahesh"}
    assert doctor_ids == expected_ids


@pytest.mark.asyncio
async def test_get_specific_doctor_public(client: AsyncClient):
    """Test getting a specific doctor by ID via public endpoint."""
    response = await client.get("/api/v1/doctors/public/dr.deven")
    assert response.status_code == 200
    data = response.json()
    assert data["doctor_id"] == "dr.deven"
    assert data["name"] == "Dr. Deven Patel"


@pytest.mark.asyncio
async def test_get_nonexistent_doctor_public(client: AsyncClient):
    """Test getting a non-existent doctor returns 404 via public endpoint."""
    response = await client.get("/api/v1/doctors/public/nonexistent")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_filter_doctors_by_specialization_public(client: AsyncClient):
    """Test filtering doctors by specialization via public endpoint."""
    response = await client.get("/api/v1/doctors/public?specialization=cardiologist")
    assert response.status_code == 200
    data = response.json()
    assert "doctors" in data
    assert len(data["doctors"]) >= 1
    # Should only return cardiologist
    assert all("cardiolog" in d["specialization"].lower() for d in data["doctors"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
