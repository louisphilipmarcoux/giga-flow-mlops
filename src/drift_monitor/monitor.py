import asyncio
import json
import logging
import os

import mlflow
import pandas as pd
from aiohttp import web
from aiokafka import AIOKafkaConsumer
from evidently import Report
from evidently.metrics import ValueDrift
from prometheus_client import REGISTRY, Gauge

logger = logging.getLogger("drift_monitor")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

logger.info("Drift Monitor Service starting...")

# --- CONFIGURATION ---
KAFKA_TOPIC = "giga-flow-messages"
KAFKA_SERVER = os.getenv("KAFKA_SERVER", "localhost:9092")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow_server:5000")
MLFLOW_MODEL_NAME = "giga-flow-sentiment"
MLFLOW_MODEL_ALIAS = "champion"
BATCH_SIZE = 100
METRICS_PORT = 8001
DRIFT_THRESHOLD = 0.05  # p-value below this = drift detected
DRIFT_RETRAIN_AFTER = int(os.getenv("DRIFT_RETRAIN_AFTER", "3"))  # consecutive drift checks before retraining
MODEL_SERVICE_URL = os.getenv("MODEL_SERVICE_URL", "http://model_service:8000")

# --- PROMETHEUS METRICS ---
DRIFT_SCORE = Gauge("sentiment_data_drift_score", "Data Drift Score (1 = Drift, 0 = No Drift)")
DRIFT_P_VALUE = Gauge("sentiment_data_drift_p_value", "Data Drift P-Value")
RETRAIN_TRIGGERED = Gauge("sentiment_retrain_triggered", "Retraining triggered (1 = yes)")

# --- DRIFT STATE ---
consecutive_drift_count = 0

# --- GLOBAL VARS ---
reference_data = None
live_data_batch = []
client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)


async def load_reference_data():
    """
    Downloads the 'training_data.parquet' artifact from the
    current 'champion' model in MLflow.
    """
    global reference_data
    logger.info(f"Connecting to MLflow at {MLFLOW_TRACKING_URI}...")

    retries = 10
    delay = 5
    while retries > 0:
        try:
            version = client.get_model_version_by_alias(MLFLOW_MODEL_NAME, MLFLOW_MODEL_ALIAS)
            logger.info(f"Found champion model: Version {version.version}, Run ID {version.run_id}")

            local_path = client.download_artifacts(
                run_id=version.run_id, path="reference_data/training_data.parquet", dst_path="."
            )
            logger.info(f"Reference data artifact downloaded to {local_path}")

            reference_data = pd.read_parquet(local_path)
            reference_data = reference_data[["text"]]
            logger.info(f"Reference data loaded: {len(reference_data)} rows.")
            break

        except Exception as e:
            logger.warning(f"Error loading reference data (retrying): {e}")
            retries -= 1
            if retries == 0:
                logger.error("FATAL: Could not load reference data. Exiting.")
                raise e
            await asyncio.sleep(delay)
            delay = min(delay * 2, 60)


async def run_drift_check(live_batch_df):
    """
    Runs an Evidently AI drift report comparing the live
    batch to the reference data.
    """
    if reference_data is None:
        logger.warning("Reference data not loaded, skipping drift check.")
        return

    global consecutive_drift_count

    logger.info(f"Running drift check on {len(live_batch_df)} new records...")
    try:
        report = Report(metrics=[ValueDrift(column="text")])
        snapshot = report.run(reference_data=reference_data, current_data=live_batch_df)

        result = list(snapshot.metric_results.values())[0]
        drift_score = float(result.value)
        drift_detected = drift_score < DRIFT_THRESHOLD
        score = 1.0 if drift_detected else 0.0

        logger.info(
            f"Drift Check: score={drift_score:.4f}, Drift={drift_detected}, Consecutive={consecutive_drift_count}"
        )

        DRIFT_SCORE.set(score)
        DRIFT_P_VALUE.set(drift_score)

        # Track consecutive drift detections
        if drift_detected:
            consecutive_drift_count += 1
            if consecutive_drift_count >= DRIFT_RETRAIN_AFTER:
                logger.warning(f"Drift sustained for {consecutive_drift_count} checks — triggering retraining!")
                await trigger_retraining()
                consecutive_drift_count = 0
        else:
            consecutive_drift_count = 0

    except Exception as e:
        logger.error(f"Error during drift check: {e}")


async def trigger_retraining():
    """Trigger model retraining via model service reload or logging."""
    import aiohttp

    RETRAIN_TRIGGERED.set(1)
    logger.info("Auto-retraining triggered due to sustained data drift.")

    # Option 1: Reload champion model (if retraining happened externally)
    try:
        async with (
            aiohttp.ClientSession() as session,
            session.post(f"{MODEL_SERVICE_URL}/reload", timeout=aiohttp.ClientTimeout(total=120)) as resp,
        ):
            if resp.status == 200:
                logger.info("Model service reloaded successfully.")
            else:
                logger.warning(f"Model reload returned {resp.status}")
    except Exception as e:
        logger.error(f"Failed to trigger reload: {e}")

    # Reset after some time
    await asyncio.sleep(60)
    RETRAIN_TRIGGERED.set(0)


async def kafka_consumer():
    """Consumes messages from Kafka and runs drift checks."""
    global live_data_batch

    consumer = None
    delay = 5
    while consumer is None:
        try:
            logger.info("Attempting to connect Kafka Consumer...")
            consumer = AIOKafkaConsumer(
                KAFKA_TOPIC,
                bootstrap_servers=KAFKA_SERVER,
                value_deserializer=lambda v: json.loads(v.decode("utf-8")),
                group_id="drift_monitor_group",
            )
            await consumer.start()
            logger.info("Kafka Consumer connected.")
        except Exception as e:
            logger.warning(f"Kafka connection failed (retrying): {e}")
            await asyncio.sleep(delay)
            delay = min(delay * 2, 60)

    try:
        async for msg in consumer:
            text = msg.value.get("text")
            if text:
                live_data_batch.append({"text": text})

            if len(live_data_batch) >= BATCH_SIZE:
                asyncio.create_task(run_drift_check(pd.DataFrame(live_data_batch)))
                live_data_batch = []
    finally:
        await consumer.stop()


async def start_metrics_server():
    """Starts the Prometheus HTTP server."""
    logger.info(f"Starting Prometheus metrics server on port {METRICS_PORT}...")
    app = web.Application()

    async def handle_metrics(request):
        from prometheus_client import generate_latest

        resp = web.Response(body=generate_latest(REGISTRY))
        resp.content_type = "text/plain"
        return resp

    async def handle_status(request):
        return web.json_response(
            {
                "drift_detected": consecutive_drift_count > 0,
                "consecutive_drift_checks": consecutive_drift_count,
                "retrain_threshold": DRIFT_RETRAIN_AFTER,
                "reference_data_loaded": reference_data is not None,
            }
        )

    app.router.add_get("/metrics", handle_metrics)
    app.router.add_get("/status", handle_status)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", METRICS_PORT)
    await site.start()
    logger.info(f"Metrics server running on http://0.0.0.0:{METRICS_PORT}")


async def main():
    await load_reference_data()

    consumer_task = asyncio.create_task(kafka_consumer())
    metrics_task = asyncio.create_task(start_metrics_server())

    await asyncio.gather(consumer_task, metrics_task)


if __name__ == "__main__":
    asyncio.run(main())
