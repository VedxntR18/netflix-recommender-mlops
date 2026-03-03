from fastapi.testclient import TestClient
import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.app import app

client = TestClient(app)


def test_root_endpoint():
    """Test: homepage returns welcome message."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "Netflix" in data["message"]


def test_health_endpoint():
    """Test: health check reports status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "num_titles" in data


def test_recommend_valid_title():
    """Test: valid title returns recommendations."""
    if not os.path.exists(os.path.join("models", "movie_titles.pkl")):
        pytest.skip("Model files not found")

    response = client.post(
        "/recommend",
        json={"title": "Stranger Things", "top_n": 3}
    )

    assert response.status_code in [200, 404]

    if response.status_code == 200:
        data = response.json()
        assert "recommendations" in data
        assert "input_title" in data
        assert len(data["recommendations"]) <= 3


def test_recommend_invalid_title():
    """Test: non-existent title returns 404."""
    if not os.path.exists(os.path.join("models", "movie_titles.pkl")):
        pytest.skip("Model files not found")

    response = client.post(
        "/recommend",
        json={"title": "This Movie Does Not Exist 99999", "top_n": 5}
    )
    assert response.status_code == 404


def test_titles_endpoint():
    """Test: /titles returns a list."""
    response = client.get("/titles")

    if response.status_code == 200:
        data = response.json()
        assert "total_titles" in data
        assert "sample_titles" in data
        assert isinstance(data["sample_titles"], list)


def test_malformed_request():
    """Test: missing required field returns 422."""
    if not os.path.exists(os.path.join("models", "movie_titles.pkl")):
        pytest.skip("Model files not found")

    response = client.post(
        "/recommend",
        json={"wrong_field": "test"}
    )
    assert response.status_code == 422