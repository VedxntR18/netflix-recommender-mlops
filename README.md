# Netflix Recommendation System — MLOps Pipeline

> An academic/portfolio MLOps project demonstrating data versioning, reproducible model training, offline evaluation, API serving, containerization, CI/CD, deployment configuration, and simulated data-drift monitoring.

## Architecture

Data → DVC Pipeline → Preprocess → TF-IDF Train (MLflow) → Evaluate + Quality Gates  
→ FastAPI → Docker → GitHub Actions → Render Configuration → Drift Monitoring

## MLOps Components

| Area | Implementation |
|---|---|
| Data / Pipeline | DVC |
| Feature Engineering / Model | TF-IDF content-based similarity |
| Experiment Tracking | MLflow |
| Evaluation | Precision@K, Recall@K, NDCG, MAP, Hit Rate, Diversity, Coverage |
| Model Serving | FastAPI |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Cloud Deployment Configuration | Render |
| Monitoring | Chi-Square + Evidently |

## Key Results

The committed evaluation artifact (`reports/evaluation_report.json`) records an offline experiment using **300 test queries** over an **8,807-title catalog**.

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

These are **offline, model-specific experiment results**. They are not claims of production recommendation quality.

## Quick Start

```bash
git clone https://github.com/VedxntR18/netflix-recommender-mlops.git
cd netflix-recommender-mlops

python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
# source venv/bin/activate

pip install -r requirements.txt

# Reproduce preprocess → train → evaluate
dvc repro

# Start the API
uvicorn api.app:app --host 0.0.0.0 --port 8000

# Optional: view MLflow runs
mlflow ui

# Optional: run simulated drift monitoring
python monitoring/monitor.py
```

## API

After running the pipeline:

```http
GET /health
```

Returns model readiness and catalog size.

```http
POST /recommend
Content-Type: application/json

{"title":"Stranger Things","top_n":5}
```

The endpoint returns ranked recommendations with cosine-similarity scores.

```http
GET /titles
```

Returns the catalog size and a sample of available titles.

Interactive API documentation is available at `/docs` when the FastAPI server is running.

## Docker

The Docker image contains the API and generated model artifacts. Because model artifacts are intentionally excluded from Git, run the pipeline before building locally:

```bash
dvc repro
docker build -t netflix-recommender:latest .
docker run --rm -p 8000:8000 netflix-recommender:latest
```

The Docker build explicitly fails if the required trained artifacts are missing.

## CI/CD

GitHub Actions runs on pushes and pull requests targeting `main`.

The pipeline:
1. Installs the dependency ranges.
2. Runs Flake8.
3. Reproduces the DVC pipeline.
4. Checks the evaluation quality gates.
5. Runs the API test suite.
6. Uploads evaluation artifacts.
7. On pushes to `main`, builds the Docker image and starts it for a health/docs smoke test.

## Monitoring

`monitoring/monitor.py` demonstrates categorical data-drift detection using Pearson's Chi-Square test and attempts supplementary analysis with Evidently.

The current monitoring workflow **simulates new production-like data** at low, medium, and high drift levels. It is an MLOps demonstration, not live production monitoring.

Generated reports include:
- per-column drift statistics
- p-values
- category counts
- drift share
- HTML summary reports

## DVC

The repository contains a three-stage DVC pipeline:

```text
preprocess → train → evaluate
```

Machine-specific DVC remotes are intentionally kept out of version control. Configure a local remote only when needed:

```bash
dvc remote add --local <name> <url>
```

The committed `.dvc/config` contains no machine-specific remote path.

## Project Structure

```text
netflix-recommender-mlops/
├── .github/workflows/ci-cd.yml
├── .dvc/
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

Generated model files and intermediate datasets are excluded from Git and recreated by the pipeline.

## Important Limitations

This project is intentionally an **academic/portfolio MLOps implementation**, not a production Netflix-like service.

- The recommender uses content-based TF-IDF similarity rather than collaborative filtering or deep learning.
- Evaluation relevance is derived from genre overlap.
- There is no real user-history, rating, click, or implicit-feedback dataset.
- Catalog coverage is approximately 5.5% in the current experiment.
- Monitoring uses simulated drift scenarios rather than live production traffic.
- The Docker image requires generated model artifacts.
- The Render file demonstrates a deployment configuration; it does not by itself prove that a public service is currently live.
- MLflow tracks training experiments; the recommender is a similarity-based system rather than a conventional estimator exposing a standard `predict()` method.

## Team

```text
23AM1070 Vedant Vaibhav Rangnekar B2
23AM1063 Vibhav Sudhir Madhavi B3
23AM1062 Vansh Dipakkumar Patel B3
23AM1159 Shrikant Lala B3

College: RAIT, Navi Mumbai
Course: CSE AIML, 3rd Year
Subject: MLOps Skill-Based Lab
```
