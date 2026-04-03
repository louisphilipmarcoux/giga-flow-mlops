# Giga-Flow MLOps

[![CI](https://github.com/Louis-Philip/giga-flow-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/Louis-Philip/giga-flow-mlops/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![MLflow 2.9](https://img.shields.io/badge/MLflow-2.9-blue.svg)](https://mlflow.org)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg)](https://docs.docker.com/compose/)

A production-ready, end-to-end MLOps pipeline for **real-time sentiment analysis** using HuggingFace DistilBERT, Kafka streaming, MLflow model registry, and comprehensive monitoring.

## Features

- **Real-time inference** -- FastAPI service consuming from Kafka with batch prediction optimization
- **Model registry** -- MLflow with automated champion promotion based on accuracy comparison
- **Data versioning** -- DVC + MinIO for reproducible datasets and training pipelines
- **Drift detection** -- Evidently AI monitors live data drift with Prometheus alerting
- **Live dashboard** -- Streamlit UI for testing predictions and viewing sentiment distribution
- **Full observability** -- Prometheus metrics, Grafana dashboards, custom prediction latency histograms
- **CI/CD** -- GitHub Actions for automated testing and manual model training/promotion
- **Container orchestration** -- 13 Docker services with health checks, restart policies, and resource limits

## Architecture

```
                    +------------------+
                    |    Producer      |
                    | (dummy data)     |
                    +--------+---------+
                             |
                             v
                    +------------------+
                    |     Kafka        |
                    | (streaming)      |
                    +--------+---------+
                             |
              +--------------+--------------+
              |                             |
              v                             v
    +------------------+          +------------------+
    |  Model Service   |          |  Drift Monitor   |
    |  (FastAPI)       |          |  (Evidently AI)  |
    +--------+---------+          +--------+---------+
             |                             |
    +--------+---------+          +--------+---------+
    |   PostgreSQL     |          |   Prometheus     |
    |   (predictions)  |          |   (metrics)      |
    +--------+---------+          +--------+---------+
             |                             |
    +--------+---------+          +--------+---------+
    |   Dashboard      |          |    Grafana       |
    |   (Streamlit)    |          |   (dashboards)   |
    +------------------+          +------------------+

    +------------------+          +------------------+
    |   MLflow Server  |          |     MinIO        |
    |   (registry)     +--------->|   (artifacts)    |
    +------------------+          +------------------+
```

### Services (13 total)

| Service | Purpose | Port |
|---------|---------|------|
| **producer** | Simulates live data feed to Kafka | - |
| **kafka** | Message broker (with Zookeeper) | 29092 |
| **mlflow_server** | Model registry & tracking UI | 5000 |
| **model_service** | FastAPI inference + Kafka consumer | 8000 |
| **postgres_db** | Predictions & MLflow backend store | 5432 |
| **minio** | S3-compatible artifact storage | 9000/9001 |
| **dashboard** | Streamlit live monitoring UI | 8501 |
| **drift_monitor** | Evidently AI data drift detection | 8001 |
| **prometheus** | Metrics collection & alerting | 9090 |
| **grafana** | Metrics visualization dashboards | 3000 |
| **kafka_exporter** | Kafka metrics for Prometheus | 9308 |
| **cadvisor** | Container resource monitoring | 8080 |

## Technology Stack

- **ML Model:** HuggingFace DistilBERT (`distilbert-base-uncased-finetuned-sst-2-english`) via PyTorch
- **Inference Service:** FastAPI with async Kafka consumer
- **Streaming:** Apache Kafka
- **Model Registry:** MLflow 2.9.2
- **Data Versioning:** DVC + MinIO
- **Database:** PostgreSQL 14
- **Dashboard:** Streamlit
- **Drift Detection:** Evidently AI
- **Monitoring:** Prometheus + Grafana + cAdvisor
- **CI/CD:** GitHub Actions
- **Orchestration:** Docker Compose

## Quick Start

### Prerequisites

- Docker & Docker Compose
- Git

### 1. Configure Environment

```bash
cp .env.example .env
# Edit .env with your desired credentials (defaults work for local dev)
```

### 2. Start All Services

```bash
docker-compose up -d --build
```

The `model_service` will restart until a model is trained. This is expected.

### 3. Train & Deploy the First Model

```bash
# Exec into the producer container
docker-compose exec -e MLFLOW_TRACKING_URI=http://mlflow_server:5000 producer /bin/bash

# Inside the container:
jupyter nbconvert --to script src/notebooks/01_model_training.ipynb
python src/notebooks/01_model_training.py
exit
```

The `model_service` will automatically pick up the "champion" model on its next retry.

### 4. Verify It's Working

```bash
# Check model service logs
docker-compose logs -f model_service

# Test the prediction endpoint
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this product!"}'

# Check the database
docker-compose exec postgres_db psql -U gigaflow -d sentiment_db \
  -c "SELECT text, sentiment_label, processed_at FROM sentiment_predictions ORDER BY processed_at DESC LIMIT 10;"
```

## Accessing Services

| Service | URL |
|---------|-----|
| Model Service API | http://localhost:8000 |
| Model Service Health | http://localhost:8000/health |
| MLflow UI | http://localhost:5000 |
| Streamlit Dashboard | http://localhost:8501 |
| Grafana Dashboards | http://localhost:3000 (admin/admin) |
| Prometheus | http://localhost:9090 |
| MinIO Console | http://localhost:9001 |

## Monitoring

### Grafana

Pre-provisioned dashboards are available at http://localhost:3000:

- **GigaFlow MLOps Overview** — Request rate, latency, drift score, Kafka lag, container resources

### Prometheus Alerts

Alert rules are configured for:
- Data drift detected (5min sustained)
- Model service down (1min)
- High error rate (>5% 5xx responses)
- Kafka consumer lag > 1000
- Drift monitor unreachable

View active alerts at http://localhost:9090/alerts.

### Drift Detection

The drift monitor compares live Kafka messages against the training data using Evidently AI. Metrics are exposed at `:8001/metrics` and scraped by Prometheus.

## CI/CD

### GitHub Actions Workflows

- **ci.yml** — Triggered on pull requests. Starts all services, runs CI-mode training (10k samples), executes pytest suite.
- **training.yml** — Manual trigger. Runs full training, promotes model if accuracy improves, restarts model_service.

### Model Promotion

The promotion script (`scripts/promote_model.py`) compares the new model's accuracy against the current champion. Promotion only happens if the new model is better.

## Data Versioning (DVC)

The IMDB dataset is tracked with DVC and stored in MinIO:

```bash
# Pull data
dvc pull

# Reproduce the training pipeline
dvc repro
```

Pipeline stages are defined in `dvc.yaml` with parameters in `params.yaml`.

## Deploying a New Model

1. Modify the training notebook or parameters
2. Re-run training (Phase 2 steps above, or trigger the `training.yml` workflow)
3. The promotion script automatically compares accuracy and promotes if better
4. Restart model_service to load the new champion:
   ```bash
   docker-compose restart model_service
   ```

## Development

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests (requires services running)
pytest tests/ --model-service-url=http://localhost:8000 --mlflow-tracking-uri=http://localhost:5000

# Install pre-commit hooks
pre-commit install
```

## Project Structure

```
giga-flow-mlops/
├── src/
│   ├── model_service/main.py      # FastAPI inference + Kafka consumer
│   ├── producer/producer.py        # Kafka data producer
│   ├── drift_monitor/monitor.py    # Evidently drift detection
│   ├── dashboard/app.py            # Streamlit UI
│   └── notebooks/                  # Training notebook
├── scripts/promote_model.py        # Model promotion logic
├── tests/                          # Pytest suite
├── grafana/                        # Dashboard provisioning
├── prometheus/                     # Prometheus config & alerts
├── postgres/                       # DB initialization
├── data/                           # DVC-tracked datasets
├── docker-compose.yml              # Production orchestration
├── docker-compose.ci.yml           # CI port overrides
├── Dockerfile                      # Multi-stage app build
├── mlflow.Dockerfile               # MLflow server
├── dashboard.Dockerfile            # Streamlit dashboard
├── dvc.yaml                        # DVC pipeline stages
├── params.yaml                     # Training parameters
├── requirements.txt                # Python dependencies (pinned)
├── requirements-inference.txt      # Inference-only dependencies
└── .pre-commit-config.yaml         # Code quality hooks
```
