# Architecture

## System Overview

GigaFlow uses a microservices architecture with 14 Docker containers communicating over a private bridge network.

```
                    +------------------+
                    |    Producer      |
                    | (120+ messages)  |
                    | (7 languages)    |
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
    |  4 ML models:    |          |  (Evidently AI)  |
    |  - Sentiment     |          |  Auto-retrain    |
    |  - Emotions      |          +--------+---------+
    |  - Language      |                   |
    |  - Toxicity      |          +--------+---------+
    +--------+---------+          |   Prometheus     |
             |                    |   (6 alerts)     |
    +--------+---------+          +--------+---------+
    |   PostgreSQL     |                   |
    |   + Redis Cache  |          +--------+---------+
    +--------+---------+          |    Grafana       |
             |                    |   (dashboards)   |
    +--------+---------+          +------------------+
    |   Dashboard      |
    |   (Streamlit)    |
    +------------------+

    +------------------+          +------------------+
    |   MLflow Server  |          |     MinIO        |
    |   (registry)     +--------->|   (S3 storage)   |
    +------------------+          +------------------+
```

## ML Pipeline

1. **Producer** sends multilingual text messages to Kafka
2. **Model Service** consumes messages and runs 4 models:
   - **Sentiment** (modernBERT-large-multilingual) - Positive/Negative/Neutral
   - **Emotions** (multilingual GoEmotions) - 28 emotion classes
   - **Language** (XLM-RoBERTa) - language identification
   - **Toxicity** (toxic-bert) - content safety screening
3. Results stored in **PostgreSQL** with Redis caching
4. **Drift Monitor** compares live data against training data
5. If drift sustained, **auto-retraining** triggers model reload

## Model Management

- Models versioned in **MLflow** with "champion" alias
- **A/B testing** splits traffic between champion and challenger
- **Hot-swap** models via `/reload` endpoint (zero downtime)
- **Promotion script** compares accuracy and auto-promotes

## Data Pipeline

- **DVC** tracks datasets in MinIO (IMDB, Amazon, Twitter)
- Training notebook evaluates across all datasets
- Metrics logged per-dataset to MLflow
