import asyncio
import json
import os
import pandas as pd
import mlflow
from aiokafka import AIOKafkaConsumer
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from aiohttp import web
from prometheus_client import Gauge, start_http_server, REGISTRY

print("Drift Monitor Service starting...")

# --- CONFIGURATION ---
KAFKA_TOPIC = "giga-flow-messages"
KAFKA_SERVER = os.getenv("KAFKA_SERVER", "localhost:9092")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow_server:5000")
MLFLOW_MODEL_NAME = "giga-flow-sentiment"
MLFLOW_MODEL_ALIAS = "champion"
BATCH_SIZE = 100  # Number of live messages to collect before checking drift
METRICS_PORT = 8001  # Port for this service's metrics endpoint

# --- PROMETHEUS METRICS ---
# We use a Gauge for the drift score
DRIFT_SCORE = Gauge("sentiment_data_drift_score", "Data Drift Score (1 = Drift, 0 = No Drift)")
DRIFT_P_VALUE = Gauge("sentiment_data_drift_p_value", "Data Drift P-Value")

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
    print(f"Connecting to MLflow at {MLFLOW_TRACKING_URI}...")
    
    retries = 10
    while retries > 0:
        try:
            # 1. Get the model version for the "champion" alias
            version = client.get_model_version_by_alias(MLFLOW_MODEL_NAME, MLFLOW_MODEL_ALIAS)
            print(f"Found champion model: Version {version.version}, Run ID {version.run_id}")
            
            # 2. Download the artifact from that run
            local_path = client.download_artifacts(
                run_id=version.run_id,
                path="reference_data/training_data.parquet",
                dst_path="."
            )
            print(f"Reference data artifact downloaded to {local_path}")
            
            # 3. Load into memory
            reference_data = pd.read_parquet(local_path)
            # We only care about the 'text' column for drift
            reference_data = reference_data[['text']]
            print(f"Reference data loaded: {len(reference_data)} rows.")
            break # Success!
            
        except Exception as e:
            print(f"Error loading reference data (retrying): {e}")
            retries -= 1
            if retries == 0:
                print("FATAL: Could not load reference data. Exiting.")
                raise e
            await asyncio.sleep(10)

async def run_drift_check(live_batch_df):
    """
    Runs an Evidently AI drift report comparing the live
    batch to the reference data.
    """
    if reference_data is None:
        print("Reference data not loaded, skipping drift check.")
        return

    print(f"Running drift check on {len(live_batch_df)} new records...")
    try:
        data_drift_report = Report(metrics=[DataDriftPreset(num_features=['text'])])
        data_drift_report.run(current_data=live_batch_df, reference_data=reference_data, column_mapping=None)
        
        # Get the drift report dictionary
        report_dict = data_drift_report.as_dict()
        
        # Extract the drift score and p-value
        # This path is specific to the DataDriftPreset
        drift_info = report_dict['metrics'][0]['result']
        p_value = drift_info['p_value']
        drift_detected = drift_info['drift_detected']
        score = 1.0 if drift_detected else 0.0

        print(f"Drift Check Complete: p-value={p_value:.4f}, Drift Detected={drift_detected}")

        # --- UPDATE PROMETHEUS METRICS ---
        DRIFT_SCORE.set(score)
        DRIFT_P_VALUE.set(p_value)

    except Exception as e:
        print(f"Error during drift check: {e}")

async def kafka_consumer():
    """
    Consumes messages from Kafka and runs drift checks.
    """
    global live_data_batch
    
    consumer = None
    while consumer is None:
        try:
            print("Attempting to connect Kafka Consumer...")
            consumer = AIOKafkaConsumer(
                KAFKA_TOPIC,
                bootstrap_servers=KAFKA_SERVER,
                value_deserializer=lambda v: json.loads(v.decode('utf-8')),
                group_id="drift_monitor_group" # Use a dedicated group
            )
            await consumer.start()
            print("Kafka Consumer connected.")
        except Exception as e:
            print(f"Kafka connection failed (retrying): {e}")
            await asyncio.sleep(5)

    try:
        async for msg in consumer:
            text = msg.value.get('text')
            if text:
                live_data_batch.append({'text': text})
            
            if len(live_data_batch) >= BATCH_SIZE:
                # Run drift check in a separate (non-blocking) task
                asyncio.create_task(run_drift_check(pd.DataFrame(live_data_batch)))
                live_data_batch = [] # Clear the batch
    finally:
        await consumer.stop()

async def start_metrics_server():
    """
    Starts the Prometheus HTTP server.
    """
    print(f"Starting Prometheus metrics server on port {METRICS_PORT}...")
    # Use aiohttp for an async-friendly web server
    app = web.Application()
    
    async def handle_metrics(request):
        # This endpoint is scraped by Prometheus
        from prometheus_client import generate_latest
        resp = web.Response(body=generate_latest(REGISTRY))
        resp.content_type = "text/plain"
        return resp

    app.router.add_get("/metrics", handle_metrics)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', METRICS_PORT)
    await site.start()
    print(f"Metrics server running on http://0.0.0.0:{METRICS_PORT}")


async def main():
    # Load the reference data from MLflow first
    await load_reference_data()
    
    # Start the other tasks
    consumer_task = asyncio.create_task(kafka_consumer())
    metrics_task = asyncio.create_task(start_metrics_server())
    
    await asyncio.gather(consumer_task, metrics_task)

if __name__ == "__main__":
    asyncio.run(main())