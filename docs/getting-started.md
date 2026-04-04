# Getting Started

## Prerequisites

- Docker & Docker Compose
- Git
- 8GB+ RAM allocated to Docker (WSL2: edit `~/.wslconfig`)

## Quick Start

```bash
# Clone and configure
git clone https://github.com/louisphilipmarcoux/giga-flow-mlops.git
cd giga-flow-mlops
cp .env.example .env

# Start all services (14 containers)
make up

# Train the first model (inside producer container)
make train

# Open the dashboard
open http://localhost:8501
```

## Services

| Service | URL | Description |
|---------|-----|-------------|
| Model Service | http://localhost:8000 | Prediction API |
| API Docs | http://localhost:8000/docs | Swagger/OpenAPI |
| MLflow UI | http://localhost:5000 | Model registry |
| Dashboard | http://localhost:8501 | Streamlit UI |
| Grafana | http://localhost:3000 | Monitoring dashboards |
| Prometheus | http://localhost:9090 | Metrics & alerts |
| MinIO Console | http://localhost:9001 | Object storage |

## First Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this product!"}'
```

Response:
```json
{
  "text": "I love this product!",
  "sentiment_label": "Positive",
  "top_emotion": "love",
  "emotions": {"love": 0.95, "admiration": 0.05, ...},
  "language": "en",
  "is_toxic": false,
  "cached": false
}
```

## Useful Commands

```bash
make up          # Start all services
make down        # Stop all services
make test        # Run test suite
make test-unit   # Run unit tests only
make lint        # Check code quality
make train       # Train a model
make logs        # Tail all logs
make load-test   # Run Locust load tests
make clean       # Stop and remove volumes
```
