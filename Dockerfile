# Dockerfile for Netflix Recommendation System
# Builds a self-contained container with the API and trained model

FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \\
    PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY api/ ./api/
COPY models/ ./models/
COPY params.yaml .

# Fail the build clearly when trained artifacts were not generated first.
RUN test -f models/tfidf_vectorizer.pkl && \
    test -f models/tfidf_matrix.pkl && \
    test -f models/movie_titles.pkl

EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]