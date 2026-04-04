<p align="center">
  <h1 align="center">Giga-Flow MLOps</h1>
  <p align="center">
    A production-grade, end-to-end MLOps pipeline for real-time sentiment analysis.<br/>
    From data streaming to model serving, monitoring, and automated retraining — all containerized.
  </p>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
  <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.11-3776AB.svg?logo=python&logoColor=white" alt="Python 3.11"></a>
  <a href="https://mlflow.org"><img src="https://img.shields.io/badge/MLflow-2.9-0194E2.svg?logo=mlflow&logoColor=white" alt="MLflow"></a>
  <a href="https://docs.docker.com/compose/"><img src="https://img.shields.io/badge/Docker-Compose-2496ED.svg?logo=docker&logoColor=white" alt="Docker"></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/Apache_Kafka-231F20?logo=apachekafka&logoColor=white" alt="Kafka">
  <img src="https://img.shields.io/badge/PostgreSQL-4169E1?logo=postgresql&logoColor=white" alt="PostgreSQL">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/HuggingFace-FFD21E?logo=huggingface&logoColor=black" alt="HuggingFace">
  <img src="https://img.shields.io/badge/Prometheus-E6522C?logo=prometheus&logoColor=white" alt="Prometheus">
  <img src="https://img.shields.io/badge/Grafana-F46800?logo=grafana&logoColor=white" alt="Grafana">
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/DVC-13ADC7?logo=dvc&logoColor=white" alt="DVC">
</p>

---

## What is this?

GigaFlow is a **complete MLOps platform** that demonstrates how to build, deploy, monitor, and continuously improve a machine learning model in production. It processes a live stream of text messages through Apache Kafka, runs real-time sentiment analysis using a HuggingFace DistilBERT transformer, stores predictions in PostgreSQL, and monitors for data drift — all orchestrated with Docker Compose.

This isn't a toy example. It implements the patterns you'd find in a real production ML system:

- **Streaming inference** with async Kafka consumers and batch prediction
- **Model versioning & promotion** with MLflow's champion/challenger pattern
- **Data drift detection** with Evidently AI and Prometheus alerting
- **Full observability** with custom metrics, pre-built Grafana dashboards, and alert rules
- **Reproducible pipelines** with DVC data versioning and parameterized training
- **CI/CD** with GitHub Actions for automated testing and one-click retraining

## Features

- **Real-time inference** — FastAPI service consuming from Kafka with batch prediction optimization
- **Model registry** — MLflow with automated champion promotion based on accuracy comparison
- **Data versioning** — DVC + MinIO for reproducible datasets and training pipelines
- **Drift detection** — Evidently AI monitors live data drift with Prometheus alerting
- **Live dashboard** — Streamlit UI for testing predictions and viewing sentiment distribution
- **Full observability** — Prometheus metrics, Grafana dashboards, custom prediction latency histograms
- **CI/CD** — GitHub Actions for automated testing and manual model training/promotion
- **Production-hardened** — Health checks, restart policies, resource limits, input validation, structured logging

## Architecture

```
                    +------------------+
                    |    Producer      |
                    | (simulated data) |
                    +--------+---------+
                             |
                             v
                    +------------------+
                    |     Kafka        |
                    |  (streaming)     |
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
    |   (registry)     +--------->|   (S3 storage)   |
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

## Tech Stack

| Category | Technology |
|----------|------------|
| **ML Model** | HuggingFace DistilBERT via PyTorch |
| **Inference** | FastAPI + async Kafka consumer |
| **Streaming** | Apache Kafka |
| **Model Registry** | MLflow 2.9 |
| **Data Versioning** | DVC + MinIO |
| **Database** | PostgreSQL 14 |
| **Dashboard** | Streamlit |
| **Drift Detection** | Evidently AI |
| **Monitoring** | Prometheus + Grafana + cAdvisor |
| **CI/CD** | GitHub Actions |
| **Orchestration** | Docker Compose (13 services) |

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
# Or use: make up
```

The `model_service` will restart until a model is trained. This is expected.

### 3. Train & Deploy the First Model

```bash
# Using Makefile
make train

# Or manually
docker-compose exec -e MLFLOW_TRACKING_URI=http://mlflow_server:5000 producer /bin/bash
jupyter nbconvert --to script src/notebooks/01_model_training.ipynb
python src/notebooks/01_model_training.py
exit
```

The `model_service` will automatically pick up the "champion" model on its next retry.

### 4. Verify It's Working

```bash
# Check model service logs
make logs-model

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
| Model Service API | `http://localhost:8000` |
| Model Service Health | `http://localhost:8000/health` |
| MLflow UI | `http://localhost:5000` |
| Streamlit Dashboard | `http://localhost:8501` |
| Grafana Dashboards | `http://localhost:3000` (admin/admin) |
| Prometheus | `http://localhost:9090` |
| MinIO Console | `http://localhost:9001` |

## Monitoring

### Grafana

Pre-provisioned dashboards are available at `http://localhost:3000`:

- **GigaFlow MLOps Overview** — Request rate, latency (p95), drift score, Kafka consumer lag, container CPU/memory

### Prometheus Alerts

Alert rules are configured for:

- Data drift detected (5min sustained)
- Model service down (1min)
- High error rate (>5% 5xx responses)
- Kafka consumer lag > 1000
- Drift monitor unreachable

View active alerts at `http://localhost:9090/alerts`.

### Drift Detection

The drift monitor consumes the same Kafka stream, batches messages (100 per check), and compares against the champion model's training data using Evidently AI. Drift metrics are exposed via Prometheus and visualized in Grafana.

## CI/CD

### GitHub Actions Workflows

| Workflow | Trigger | Description |
|----------|---------|-------------|
| **CI** | Pull request | Starts all services, runs CI-mode training (10k samples), executes pytest |
| **Lint** | Push / PR | Runs ruff linter and formatter checks |
| **Training** | Manual | Full training, smart promotion, model_service restart |

### Model Promotion

The promotion script (`scripts/promote_model.py`) compares the new model's accuracy against the current champion. Promotion only happens if the new model is better — no manual intervention needed.

## Data Versioning (DVC)

The IMDB dataset is tracked with DVC and stored in MinIO:

```bash
dvc pull          # Pull data
dvc repro         # Reproduce the training pipeline
```

Pipeline stages are defined in `dvc.yaml` with parameters in `params.yaml`.

## Deploying a New Model

1. Modify the training notebook or parameters in `params.yaml`
2. Re-run training (`make train` or trigger the Training workflow)
3. The promotion script automatically compares accuracy and promotes if better
4. Restart model_service to load the new champion:
   ```bash
   docker-compose restart model_service
   ```

## Development

```bash
make install      # Install dev dependencies + pre-commit hooks
make test-unit    # Run unit tests (no Docker needed)
make test         # Run full test suite (requires services)
make lint         # Check code with ruff
make lint-fix     # Auto-fix lint issues
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for full development guidelines.

## Project Structure

```
giga-flow-mlops/
├── src/
│   ├── model_service/main.py      # FastAPI: sentiment, emotions, language, toxicity
│   ├── producer/producer.py        # Kafka producer (120+ multilingual messages)
│   ├── drift_monitor/monitor.py    # Drift detection + auto-retraining
│   ├── dashboard/app.py            # Streamlit UI (test, dashboard, model registry)
│   └── notebooks/                  # Training notebook (dual-model architecture)
├── scripts/promote_model.py        # Model promotion logic
├── tests/                          # Pytest + Locust load tests
│   └── locustfile.py               # Load testing configuration
├── k8s/                            # Kubernetes deployment manifests
├── .github/workflows/              # CI, Lint, Training workflows
├── grafana/                        # Dashboard provisioning
├── prometheus/                     # Config & alert rules
├── data/                           # DVC-tracked datasets
├── docker-compose.yml              # Production orchestration (13 services)
├── Dockerfile                      # Multi-stage app build (inference)
├── training.Dockerfile             # Training container
├── base.Dockerfile                 # Pre-built base image
├── Makefile                        # Development commands
├── pyproject.toml                  # Project config (ruff, pytest)
├── dvc.yaml                        # Reproducible pipeline
├── params.yaml                     # Training parameters
└── requirements.txt                # Pinned dependencies
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
