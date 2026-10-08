# Netflix Recommendation System — MLOps Pipeline

> An end-to-end MLOps mini-project demonstrating data versioning, model training, evaluation, API serving, containerization, CI/CD, deployment configuration, and data-drift monitoring.

## Architecture

Data (DVC) → Preprocess → Train (MLflow) → Evaluate (Metrics + Quality Gates)
→ API (FastAPI) → Container (Docker) → CI/CD (GitHub Actions)
→ Deploy (Render) → Monitor (Evidently)


## Tech Stack

| Component | Tool |
|---|---|
| Data Versioning | DVC |
| Experiment Tracking | MLflow |
| Model Evaluation | Precision@K, Recall@K, NDCG@K, MAP, Hit Rate, Diversity |
| Model Serving | FastAPI |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Cloud Deployment | Render.com |
| Monitoring | Evidently AI |

## Evaluation Metrics

| Metric | Score |
|---|---|
The committed evaluation artifact (`reports/evaluation_report.json`) currently records results from 300 test queries over 8,807 catalog items:

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

The repository includes a Render deployment configuration in `render.yaml`. A previously listed public endpoint is not treated as a confirmed live service until it is independently verified.


## Project Structure
```text
netflix-recommender-mlops/
├── .github/workflows/ci-cd.yml    # CI/CD pipeline
├── api/app.py                     # FastAPI REST API
├── src/preprocess.py              # Data cleaning
├── src/train.py                   # Model training + MLflow
├── src/evaluate.py                # Model evaluation + quality gates
├── monitoring/monitor.py          # Data drift detection
├── tests/test_api.py              # Automated tests
├── Dockerfile                     # Container definition
├── dvc.yaml                       # DVC pipeline (3 stages)
├── params.yaml                    # Configuration
└── requirements.txt               # Dependencies
```

## Important Limitations

This is an academic/portfolio MLOps project rather than a production Netflix-like recommendation service.

- The recommender is content-based TF-IDF rather than collaborative or deep-learning based.
- Evaluation relevance is derived from genre overlap.
- There is no real user-history or implicit-feedback dataset.
- Monitoring uses simulated production data.
- The serving image requires generated model artifacts; a completely fresh clone is not a ready-to-serve image until the pipeline is run.
- Cloud deployment configuration is included, but a live public endpoint should not be assumed to be available unless independently verified.

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