# Model Registry

## Current Models

| Model | Purpose | Languages | Accuracy |
|-------|---------|-----------|----------|
| `clapAI/modernBERT-large-multilingual-sentiment` | Sentiment classification | 16+ | 98.5% (IMDB) |
| `AnasAlokla/multilingual_go_emotions_V1.1` | Emotion detection (28 classes) | 6+ | 88.6% (GoEmotions) |
| `papluca/xlm-roberta-base-language-detection` | Language identification | 20+ | N/A |
| `unitary/toxic-bert` | Toxicity screening | English | N/A |

## Emotion Classes (28)

**Positive:** admiration, amusement, approval, caring, desire, excitement, gratitude, joy, love, optimism, pride, relief

**Negative:** anger, annoyance, disappointment, disapproval, disgust, embarrassment, fear, grief, nervousness, remorse, sadness

**Neutral:** neutral, realization, confusion, curiosity, surprise

## Model Versioning

Models are versioned in MLflow. The "champion" alias points to the best performing model.

```bash
# View models
open http://localhost:5000

# Promote a model
docker compose exec producer python scripts/promote_model.py

# Hot-swap in production
curl -X POST http://localhost:8000/reload
curl -X POST http://localhost:8000/reload -d '{"version": 5}'
```

## A/B Testing

```bash
# Start A/B test: 20% traffic to version 3
curl -X POST http://localhost:8000/ab-test \
  -H "Content-Type: application/json" \
  -d '{"challenger_version": 3, "split": 0.2}'

# Check status
curl http://localhost:8000/ab-test

# Stop A/B test
curl -X DELETE http://localhost:8000/ab-test
```

## Training a New Model

```bash
# CI mode (fast, 2K samples)
make train

# Full training (50K samples)
docker compose exec -e TRAINING_MODE=FULL producer ...

# Try a different model
docker compose exec -e HF_MODEL_NAME=textattack/bert-base-uncased-imdb producer ...
```

## Tested Models

| Version | Model | Accuracy | Notes |
|---------|-------|----------|-------|
| V1 | distilbert-sst-2 | 89.6% | English only, fast |
| V3 | lvwerra/distilbert-imdb | 95.0% | English, IMDB-tuned |
| V5 | textattack/bert-imdb | 95.5% | English, BERT-based |
| V9 | nlptown/bert-multilingual | 90.5% | 6 languages, star ratings |
| V10 | modernBERT-large-multilingual | 98.5% | 16+ languages, best overall |
