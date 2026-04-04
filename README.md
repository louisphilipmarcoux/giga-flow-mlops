<p align="center">
  <h1 align="center">GigaFlow MLOps</h1>
  <p align="center">
    A production-grade, end-to-end MLOps platform for real-time multilingual<br/>
    sentiment analysis, emotion detection, language identification, and toxicity screening.
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
  <img src="https://img.shields.io/badge/Redis-DC382D?logo=redis&logoColor=white" alt="Redis">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/HuggingFace-FFD21E?logo=huggingface&logoColor=black" alt="HuggingFace">
  <img src="https://img.shields.io/badge/Prometheus-E6522C?logo=prometheus&logoColor=white" alt="Prometheus">
  <img src="https://img.shields.io/badge/Grafana-F46800?logo=grafana&logoColor=white" alt="Grafana">
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Kubernetes-326CE5?logo=kubernetes&logoColor=white" alt="Kubernetes">
</p>

---

## What is this?

GigaFlow is a **complete MLOps platform** that processes a live stream of text through Apache Kafka, runs **four ML models** simultaneously (sentiment, emotions, language detection, toxicity), stores predictions in PostgreSQL with Redis caching, and monitors for data drift with automated retraining.

This isn't a toy example. It implements production patterns:

- **4 ML models** running per prediction (sentiment, 28 emotions, language, toxicity)
- **98.5% sentiment accuracy** with multilingual support (16+ languages)
- **A/B testing** with configurable traffic splitting between model versions
- **Model explainability** with word-level importance highlighting
- **User feedback loop** with correction tracking for continuous improvement
- **Auto-retraining** triggered by sustained data drift detection
- **API authentication**, **rate limiting** (60 req/min), and **Redis caching**
- **Batch prediction** endpoint (up to 100 texts per request)
- **Zero-downtime model deployment** via hot-swap from the dashboard
- **Full observability** with Prometheus metrics, Grafana dashboards, and 6 alert rules
- **14 Docker services** with health checks, restart policies, and resource limits

## Architecture

```
Producer (120+ multilingual messages)
    |
    v
Kafka (streaming)
    |
    +---> Model Service (4 ML models) ---> PostgreSQL + Redis
    |         |
    |         +---> Streamlit Dashboard
    |
    +---> Drift Monitor (Evidently AI) ---> Prometheus ---> Grafana
    
MLflow (model registry) <---> MinIO (S3 artifacts)
```

## Quick Start

```bash
git clone https://github.com/louisphilipmarcoux/giga-flow-mlops.git
cd giga-flow-mlops
cp .env.example .env
make up              # Start 14 services
make train           # Train first model
```

Then open:
- **Dashboard:** http://localhost:8501
- **API Docs:** http://localhost:8000/docs
- **MLflow:** http://localhost:5000
- **Grafana:** http://localhost:3000 (admin/admin)

## API

```bash
# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I absolutely love this product!"}'

# Batch prediction (up to 100 texts)
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Great product!", "Terrible service.", "C est correct."]}'

# Explain prediction (word importance)
curl -X POST http://localhost:8000/explain \
  -H "Content-Type: application/json" \
  -d '{"text": "The product is amazing but shipping was terrible"}'
```

**Response:**
```json
{
  "sentiment_label": "Positive",
  "top_emotion": "love",
  "emotions": {"love": 0.95, "admiration": 0.05, ...},
  "language": "en",
  "toxicity_score": 0.0007,
  "is_toxic": false,
  "cached": false
}
```

Full API docs at http://localhost:8000/docs (Swagger UI).

## Services (14)

| Service | Port | Description |
|---------|------|-------------|
| model_service | 8000 | FastAPI with 4 ML models |
| kafka | 29092 | Message streaming |
| postgres_db | 5432 | Predictions + MLflow store |
| redis | 6379 | Prediction cache |
| mlflow_server | 5000 | Model registry & tracking |
| minio | 9000 | S3-compatible storage |
| dashboard | 8501 | Streamlit live UI |
| drift_monitor | 8001 | Data drift detection |
| prometheus | 9090 | Metrics & alerting |
| grafana | 3000 | Visualization dashboards |
| producer | - | Multilingual data feed |
| kafka_exporter | 9308 | Kafka metrics |
| cadvisor | 8080 | Container monitoring |
| zookeeper | 2181 | Kafka coordination |

## ML Models

| Model | Task | Languages | Metric |
|-------|------|-----------|--------|
| modernBERT-large-multilingual | Sentiment | 16+ | 98.5% accuracy |
| multilingual_go_emotions | 28 Emotions | 6+ | 88.6% accuracy |
| xlm-roberta-language-detection | Language ID | 20+ | - |
| toxic-bert | Toxicity | English | - |

## Features

### Model Management
- **Model Registry** with MLflow versioning and champion alias
- **A/B Testing** — split traffic between champion and challenger
- **Hot-Swap** — reload models via API or dashboard (zero downtime)
- **Model Explainability** — word-level importance highlighting

### Data Pipeline
- **Kafka streaming** with async consumer
- **DVC** data versioning (IMDB, Amazon, Twitter datasets)
- **Cross-dataset evaluation** (accuracy per dataset in MLflow)
- **Drift detection** with auto-retraining on sustained drift

### API
- **Authentication** via API key (X-API-Key header)
- **Rate limiting** (60/min predict, 10/min batch)
- **Redis caching** (1 hour TTL, cache hits return instantly)
- **Batch prediction** (up to 100 texts per request)
- **User feedback** with correction tracking

### Monitoring
- **Prometheus** metrics (latency, throughput, drift, emotions, languages)
- **Grafana** dashboards (auto-provisioned)
- **6 alert rules** (drift, service down, error rate, lag, retraining)
- **Streamlit dashboard** with live predictions, charts, model registry

### Infrastructure
- **14 Docker services** with health checks and restart policies
- **Kubernetes manifests** with HPA auto-scaling
- **Pre-built base images** on Docker Hub (1-second rebuilds)
- **Locust load testing** configuration
- **GitHub Actions** CI/CD (lint, test, train)

## Development

```bash
make install      # Install deps + pre-commit hooks
make test-unit    # Unit tests (no Docker needed)
make test         # Full test suite
make lint         # Ruff check + format
make load-test    # Locust load testing
```

## Documentation

Full docs: https://louisphilipmarcoux.github.io/giga-flow-mlops

- [Getting Started](docs/getting-started.md)
- [API Reference](docs/api-reference.md)
- [Architecture](docs/architecture.md)
- [Models](docs/models.md)
- [Monitoring](docs/monitoring.md)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## License

MIT License — see [LICENSE](LICENSE) for details.
