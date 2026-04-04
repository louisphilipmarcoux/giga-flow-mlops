# API Reference

Base URL: `http://localhost:8000`

Interactive docs: `http://localhost:8000/docs` (Swagger UI)

## Authentication

Set the `API_KEYS` environment variable to enable authentication. Requests must include the `X-API-Key` header.

```bash
# Without auth (default)
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"text": "hello"}'

# With auth
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"text": "hello"}'
```

## Rate Limits

| Endpoint | Limit |
|----------|-------|
| `POST /predict` | 60/minute per IP |
| `POST /predict/batch` | 10/minute per IP |

## Endpoints

### `POST /predict`

Analyze a single text for sentiment, emotions, language, and toxicity.

**Request:**
```json
{"text": "I absolutely love this product!"}
```

**Response:**
```json
{
  "text": "I absolutely love this product!",
  "sentiment_label": "Positive",
  "sentiment_score": 1.0,
  "top_emotion": "love",
  "emotions": {
    "love": 0.9524,
    "approval": 0.0524,
    "neutral": 0.0517,
    "curiosity": 0.0512,
    "admiration": 0.0511
  },
  "language": "en",
  "toxicity_score": 0.0007,
  "is_toxic": false,
  "model_variant": "champion",
  "cached": false
}
```

### `POST /predict/batch`

Analyze up to 100 texts in a single request.

**Request:**
```json
{
  "texts": [
    "I love this!",
    "This is terrible.",
    "C'est fantastique!"
  ]
}
```

**Response:**
```json
{
  "predictions": [...],
  "count": 3
}
```

### `POST /explain`

Get word-level importance highlighting for a prediction.

**Request:**
```json
{"text": "The product is amazing but the shipping was terrible"}
```

**Response:**
```json
{
  "text": "The product is amazing but the shipping was terrible",
  "sentiment": "Positive",
  "explanation": [
    {"word": "The", "importance": 0.0},
    {"word": "product", "importance": 0.02},
    {"word": "amazing", "importance": 0.45},
    {"word": "terrible", "importance": -0.38}
  ]
}
```

### `POST /feedback`

Submit feedback on a prediction.

```json
{
  "original_text": "Some text",
  "original_sentiment": "Positive",
  "original_emotion": "joy",
  "corrected_sentiment": "Neutral",
  "is_correct": false
}
```

### `GET /feedback/stats`

Get feedback statistics.

### `POST /reload`

Hot-swap the model without restarting the service.

```json
{"version": 5}
```

Or load the champion: `POST /reload` with empty body.

### `POST /ab-test`

Set up A/B testing between champion and challenger.

```json
{
  "challenger_version": 3,
  "split": 0.2
}
```

### `GET /health`

Health check with dependency status.

### `GET /model`

Currently loaded model info.

### `GET /metrics`

Prometheus metrics endpoint.
