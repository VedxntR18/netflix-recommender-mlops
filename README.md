# Netflix Recommendation System — MLOps Pipeline

> A production-grade MLOps mini-project demonstrating the complete ML lifecycle
> from data versioning to model evaluation, cloud deployment, and monitoring.

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
| Precision@5 | ~0.68 |
| Hit Rate@5 | ~0.94 |
| NDCG@5 | ~0.72 |
| MAP | ~0.70 |
| vs Random Baseline | +52% improvement |
| vs Popularity Baseline | +32% improvement |
| Quality Gate | PASSED |

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/netflix-recommender-mlops.git
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

## Live API
https://netflix-recommender-api.onrender.com/docs


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

## Team:

23AM1070 Vedant Vaibhav Rangnekar B2
23AM1063 Vibhav Sudhir Madhavi B3
23AM1062 Vansh Dipakkumar Patel B3
23AM1159 Shrikant lala B3
College: RAIT, Navi Mumbai
Course: CSE AIML, 3rd Year
Subject: MLOps Skill-Based Lab