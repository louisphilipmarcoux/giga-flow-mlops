# GigaFlow MLOps Documentation

GigaFlow is a production-grade, end-to-end MLOps platform for real-time sentiment analysis, emotion detection, language identification, and toxicity screening. It processes live text streams through Apache Kafka, runs multi-model inference, and stores results with full observability.

## Quick Links

- [Getting Started](getting-started.md)
- [API Reference](api-reference.md)
- [Architecture](architecture.md)
- [Model Registry](models.md)
- [Monitoring](monitoring.md)

## Key Features

- **Dual-model sentiment + emotion analysis** (98.5% accuracy, 28 emotions)
- **Multilingual support** (16+ languages for sentiment, 6+ for emotions)
- **Language detection** and **toxicity screening**
- **A/B testing** with configurable traffic splitting
- **Model explainability** with word-level importance highlighting
- **User feedback loop** for continuous improvement
- **Auto-retraining** triggered by sustained data drift
- **API authentication**, **rate limiting**, and **Redis caching**
- **Batch prediction** endpoint (up to 100 texts per request)
- **Real-time streaming** from Kafka with live Streamlit dashboard
- **Full observability** with Prometheus metrics and Grafana dashboards

## Tech Stack

| Component | Technology |
|-----------|------------|
| ML Models | HuggingFace Transformers (ModernBERT, RoBERTa, XLM-R) |
| Inference | FastAPI + async Kafka consumer |
| Streaming | Apache Kafka |
| Model Registry | MLflow 2.9 |
| Data Versioning | DVC + MinIO |
| Database | PostgreSQL 14 |
| Caching | Redis 7 |
| Dashboard | Streamlit |
| Drift Detection | Evidently AI |
| Monitoring | Prometheus + Grafana |
| Container Orchestration | Docker Compose / Kubernetes |
| CI/CD | GitHub Actions |
