import sys
from pathlib import Path

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from api.app import app  # noqa: E402

client = TestClient(app)

MODEL_FILES = [
    PROJECT_ROOT / "models" / "tfidf_vectorizer.pkl",
    PROJECT_ROOT / "models" / "tfidf_matrix.pkl",
    PROJECT_ROOT / "models" / "movie_titles.pkl",
]
MODEL_READY = all(path.exists() for path in MODEL_FILES)


def test_root_endpoint():
    """Test: homepage returns welcome message."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "Netflix" in data["message"]


def test_health_endpoint():
    """Test: health reports model readiness accurately."""
    response = client.get("/health")

    if MODEL_READY:
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["num_titles"] > 0
    else:
        assert response.status_code == 503


def test_recommend_valid_title():
    """Test: valid title returns recommendations when artifacts exist."""
    response = client.post(
        "/recommend",
        json={"title": "Stranger Things", "top_n": 3},
    )

    if not MODEL_READY:
        assert response.status_code == 503
        return

    assert response.status_code == 200
    data = response.json()
    assert data["input_title"].lower() == "stranger things"
    assert data["status"] == "success"
    assert 0 < len(data["recommendations"]) <= 3
    assert all("title" in item and "similarity_score" in item for item in data["recommendations"])


def test_recommend_invalid_title():
    """Test: non-existent title returns 404 when model is ready."""
    response = client.post(
        "/recommend",
        json={"title": "This Movie Does Not Exist 99999", "top_n": 5},
    )

    if not MODEL_READY:
        assert response.status_code == 503
        return

    assert response.status_code == 404


def test_titles_endpoint():
    """Test: /titles returns catalog samples when model is ready."""
    response = client.get("/titles")

    if not MODEL_READY:
        assert response.status_code == 503
        return

    assert response.status_code == 200
    data = response.json()
    assert data["total_titles"] > 0
    assert isinstance(data["sample_titles"], list)
    assert 0 < len(data["sample_titles"]) <= 20


def test_malformed_request():
    """Test: missing required field returns 422."""
    response = client.post(
        "/recommend",
        json={"wrong_field": "test"},
    )
    assert response.status_code == 422


def test_blank_title():
    """Test: whitespace-only titles are rejected."""
    response = client.post(
        "/recommend",
        json={"title": "   ", "top_n": 5},
    )
    assert response.status_code == 422


def test_top_n_bounds():
    """Test: top_n is constrained to the supported range."""
    for value in [0, 21, -1]:
        response = client.post(
            "/recommend",
            json={"title": "Stranger Things", "top_n": value},
        )
        assert response.status_code == 422
