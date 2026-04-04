# Monitoring

## Grafana Dashboards

Access: `http://localhost:3000` (admin/admin)

The **GigaFlow MLOps Overview** dashboard includes:
- Request rate and p95 latency
- Prediction latency (p50/p95)
- Predictions total and by sentiment
- Top emotions distribution
- Kafka consumer lag
- Data drift score
- Language distribution
- Content safety metrics

## Prometheus Metrics

Access: `http://localhost:9090`

### Custom Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `prediction_latency_seconds` | Histogram | Inference time per prediction |
| `prediction_input_length_chars` | Histogram | Input text length distribution |
| `predictions_total` | Counter | Predictions by sentiment label |
| `emotions_total` | Counter | Predictions by emotion |
| `languages_total` | Counter | Predictions by language |
| `toxic_predictions_total` | Counter | Toxic content detected |
| `ab_predictions_total` | Counter | A/B test predictions by variant |
| `sentiment_data_drift_score` | Gauge | Drift detection (1=drift, 0=no drift) |
| `sentiment_retrain_triggered` | Gauge | Auto-retraining triggered |

## Alert Rules

| Alert | Condition | Severity |
|-------|-----------|----------|
| DataDriftDetected | Drift score = 1 for 5min | Warning |
| ModelServiceDown | Service unreachable for 1min | Critical |
| HighErrorRate | >5% 5xx errors for 5min | Warning |
| KafkaConsumerLagHigh | Lag > 1000 for 5min | Warning |
| AutoRetrainingTriggered | Retraining fired | Info |
| DriftMonitorDown | Monitor unreachable for 2min | Warning |

## Drift Detection

The drift monitor:
1. Consumes Kafka messages in batches of 100
2. Compares against champion model's training data using Evidently AI
3. If drift detected for 3 consecutive checks, triggers auto-retraining
4. Exposes metrics at `:8001/metrics`

## Health Checks

```bash
# Model service (includes DB, model, Kafka status)
curl http://localhost:8000/health

# Drift monitor status
curl http://localhost:8001/status

# Feedback stats
curl http://localhost:8000/feedback/stats
```
