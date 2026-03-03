"""
app.py - FastAPI REST API for the Netflix Recommendation System.

What this script does:
1. Loads saved model artifacts
2. Creates a web server with endpoints (URLs)
3. When someone sends a movie title, returns top-N recommendations

What is a REST API?
- Your model becomes a web service that any app/website can call
- POST /recommend = send a title, get recommendations back
- GET /health = check if the server is alive
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# ── Initialize FastAPI app ──
app = FastAPI(
    title="Netflix Recommendation System API",
    description="MLOps-powered content-based recommendation engine",
    version="1.0.0"
)


# ── Define data models ──
class RecommendationRequest(BaseModel):
    """What the user sends to the API."""
    title: str
    top_n: int = 5


class RecommendationItem(BaseModel):
    """A single recommendation."""
    rank: int
    title: str
    similarity_score: float


class RecommendationResponse(BaseModel):
    """What the API sends back."""
    input_title: str
    recommendations: list[RecommendationItem]
    status: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    num_titles: int


# ── Load model artifacts at startup ──
MODELS_DIR = "models"


def load_model_artifacts():
    """Load saved model files from disk."""
    try:
        tfidf_vectorizer = joblib.load(
            os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
        )
        tfidf_matrix = joblib.load(
            os.path.join(MODELS_DIR, "tfidf_matrix.pkl")
        )
        movie_titles = joblib.load(
            os.path.join(MODELS_DIR, "movie_titles.pkl")
        )
        return tfidf_vectorizer, tfidf_matrix, movie_titles
    except FileNotFoundError as e:
        print(f"ERROR: Model files not found: {e}")
        return None, None, None


tfidf_vectorizer, tfidf_matrix, movie_titles = load_model_artifacts()


# ── API Endpoints ──

@app.get("/", response_model=dict)
def root():
    """Homepage of the API."""
    return {
        "message": "Welcome to the Netflix Recommendation System API",
        "docs_url": "/docs",
        "health_url": "/health",
        "recommend_url": "/recommend"
    }


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Reports if the model is loaded and ready."""
    return HealthResponse(
        status="healthy" if movie_titles is not None else "unhealthy",
        model_loaded=movie_titles is not None,
        num_titles=len(movie_titles) if movie_titles is not None else 0
    )


@app.post("/recommend", response_model=RecommendationResponse)
def get_recommendations(request: RecommendationRequest):
    """
    Main recommendation endpoint.
    Send a movie title, get back top-N similar shows.
    """
    if movie_titles is None or tfidf_matrix is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run the training pipeline first."
        )

    # Case-insensitive search
    title_lower = request.title.strip().lower()
    titles_lower = [t.lower() for t in movie_titles]

    if title_lower not in titles_lower:
        partial_matches = [
            t for t in movie_titles if title_lower in t.lower()
        ]
        if partial_matches:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Title '{request.title}' not found. "
                    f"Did you mean: {partial_matches[:5]}?"
                )
            )
        raise HTTPException(
            status_code=404,
            detail=f"Title '{request.title}' not found in the database."
        )

    title_idx = titles_lower.index(title_lower)

    sim_scores = cosine_similarity(
        tfidf_matrix[title_idx:title_idx + 1],
        tfidf_matrix
    ).flatten()

    top_n = min(request.top_n, len(movie_titles) - 1)
    sorted_indices = sim_scores.argsort()[::-1]

    recommendations = []
    rank = 1
    for idx in sorted_indices:
        if idx == title_idx:
            continue
        if rank > top_n:
            break
        recommendations.append(
            RecommendationItem(
                rank=rank,
                title=movie_titles[idx],
                similarity_score=round(float(sim_scores[idx]), 4)
            )
        )
        rank += 1

    return RecommendationResponse(
        input_title=movie_titles[title_idx],
        recommendations=recommendations,
        status="success"
    )


@app.get("/titles", response_model=dict)
def list_titles():
    """Returns all available titles for testing."""
    if movie_titles is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    return {
        "total_titles": len(movie_titles),
        "sample_titles": movie_titles[:20],
        "note": "Use /recommend endpoint with any of these titles"
    }