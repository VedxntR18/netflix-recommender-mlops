# Netflix Recommendation System — MLOps Pipeline

> An end-to-end MLOps mini-project demonstrating data versioning, model training, evaluation, API serving, containerization, CI/CD, deployment configuration, and data-drift monitoring.

## Architecture

Data (DVC) → Preprocess → Train (MLflow) → Evaluate (Metrics + Quality Gates)
→ API (FastAPI) → Container (Docker) → CI/CD (GitHub Actions)
→ Deploy (Render) → Monitor (Evidently)


## MLOps Components

| Area | Implementation |
|---|---|
| Data / Pipeline | DVC |
| Experiment Tracking | MLflow |
| Evaluation | Precision@K, Recall@K, NDCG, MAP, Hit Rate, Diversity, Coverage |
| Model Serving | FastAPI |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Cloud Deployment | Render.com |
| Monitoring | Evidently + Chi-Square |

## Key Results

The committed evaluation artifact (`reports/evaluation_report.json`) records results from 300 test queries over an 8,807-title catalog.

| Metric | Score |
|---|---:|
| Precision@5 | 0.8260 |
| Hit Rate@5 | 0.9700 |
| NDCG@5 | 0.9116 |
| MAP | 0.8463 |
| Mean Diversity@5 | 0.7849 |
| Catalog Coverage | 0.0551 |
| vs Random Precision@5 | +230.4% |
| vs Popularity Precision@5 | +74.5% |
| Quality Gate | PASSED |

These metrics are model-specific offline experiment results, not a claim of production performance.

## Quick Start

```bash
git clone https://github.com/VedxntR18/netflix-recommender-mlops.git
cd netflix-recommender-mlops

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt

# Run the full pipeline
dvc repro

# Start the API
uvicorn api.app:app --host 0.0.0.0 --port 8000

# View MLflow dashboard
mlflow ui

# Run monitoring
python monitoring/monitor.py
```

## Deployment

The repository includes a Render deployment configuration in `render.yaml`. A public endpoint is not treated as confirmed unless independently verified.


## Project Structure
```text
netflix-recommender-mlops/
├── .github/workflows/ci-cd.yml
├── api/app.py
├── data/netflix_titles.csv
├── models/.gitkeep
├── monitoring/monitor.py
├── reports/evaluation_report.json
├── src/preprocess.py
├── src/train.py
├── src/evaluate.py
├── tests/test_api.py
├── Dockerfile
├── dvc.yaml
├── dvc.lock
├── params.yaml
├── render.yaml
└── requirements.txt
```

Generated model files are intentionally excluded from Git and are produced by `dvc repro` before API/Docker execution.

## Important Limitations

This is an **academic/portfolio MLOps project**, not a production Netflix-like recommendation service.

- The recommender is content-based TF-IDF rather than collaborative filtering or deep learning.
- Evaluation relevance is derived from genre overlap.
- There is no real user-history, rating, click, or implicit-feedback dataset.
- Catalog coverage is approximately 5.5% in the current experiment.
- Monitoring uses simulated drift scenarios rather than live production traffic.
- The Docker image requires generated model artifacts.
- The Render configuration demonstrates deployment workflow but does not imply a currently live public service.
- MLflow tracks the experiment pipeline; the recommender is a similarity-based system rather than a conventional estimator with a standard `predict()` method.


This is an academic/portfolio MLOps project rather than a production Netflix-like recommendation service.

- The recommender is content-based TF-IDF rather than collaborative or deep-learning based.
- Evaluation relevance is derived from genre overlap.
- There is no real user-history or implicit-feedback dataset.
- Monitoring uses simulated production data.
- The serving image requires generated model artifacts; a completely fresh clone is not a ready-to-serve image until the pipeline is run.
- Cloud deployment configuration is included, but a live public endpoint should not be assumed to be available unless independently verified.

## API Example

After running the pipeline and starting FastAPI:

```http
POST /recommend
Content-Type: application/json

{"title":"Stranger Things","top_n":5}
```

The API returns ranked recommendations with cosine-similarity scores. `/health` returns HTTP 503 when the model artifacts are unavailable.

## Docker

```bash
dvc repro
docker build -t netflix-recommender:latest .
docker run --rm -p 8000:8000 netflix-recommender:latest
```

The Docker build intentionally fails when required trained artifacts are missing.

## Team:
```
23AM1070 Vedant Vaibhav Rangnekar B2
23AM1063 Vibhav Sudhir Madhavi B3
23AM1062 Vansh Dipakkumar Patel B3
23AM1159 Shrikant lala B3
College: RAIT, Navi Mumbai
Course: CSE AIML, 3rd Year
Subject: MLOps Skill-Based Lab
```