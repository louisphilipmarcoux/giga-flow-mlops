import asyncio
import datetime
import json
import logging
import os
import time as _time
from contextlib import asynccontextmanager

import mlflow
import pandas as pd
import sqlalchemy
from aiokafka import AIOKafkaConsumer
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram
from pydantic import BaseModel, field_validator
from sqlalchemy import Column, DateTime, Float, Integer, String
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from starlette_prometheus import PrometheusMiddleware, metrics

logger = logging.getLogger("model_service")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

# --- Custom Prometheus Metrics ---
PREDICTION_LATENCY = Histogram("prediction_latency_seconds", "Prediction inference latency in seconds")
PREDICTION_INPUT_LENGTH = Histogram(
    "prediction_input_length_chars",
    "Input text length in characters",
    buckets=[10, 50, 100, 200, 500, 1000, 2000, 5000],
)
PREDICTIONS_TOTAL = Counter("predictions_total", "Total predictions made", ["sentiment"])

# --- Pydantic Models ---
MAX_TEXT_LENGTH = 10_000


class PredictionRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty_or_too_long(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("text must not be empty")
        if len(v) > MAX_TEXT_LENGTH:
            raise ValueError(f"text must not exceed {MAX_TEXT_LENGTH} characters")
        return v


class KafkaMessage(BaseModel):
    text: str
    timestamp: float


# --- Configuration ---
KAFKA_TOPIC = "giga-flow-messages"
KAFKA_SERVER = os.getenv("KAFKA_SERVER", "localhost:9092")
MLFLOW_MODEL_NAME = "giga-flow-sentiment"
MLFLOW_MODEL_ALIAS = "champion"

# Load from environment variable, with our local DB as a fallback
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://gigaflow:password@localhost:5432/sentiment_db")

# --- SQLAlchemy Setup ---
engine = create_async_engine(DATABASE_URL, echo=False, pool_size=10, max_overflow=20, pool_pre_ping=True)
Base = declarative_base()
AsyncSessionLocal = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)


# --- Define Predictions Table (ORM Model) ---
class SentimentPrediction(Base):
    __tablename__ = "sentiment_predictions"
    id = Column(Integer, primary_key=True, index=True)
    text = Column(String, index=True)
    sentiment_score = Column(Float)
    sentiment_label = Column(String)
    message_timestamp = Column(DateTime)
    processed_at = Column(DateTime, default=datetime.datetime.utcnow)


# --- Model & Consumer Globals ---
model = None
kafka_consumer_task = None


async def create_db_and_tables():
    """Create the table in the database."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


@asynccontextmanager
async def lifespan(app):
    """Manage startup and shutdown lifecycle."""
    global model, kafka_consumer_task

    logger.info("Creating database tables...")
    await create_db_and_tables()
    logger.info("Database tables created.")

    logger.info("Loading MLflow model...")
    model_uri = f"models:/{MLFLOW_MODEL_NAME}@{MLFLOW_MODEL_ALIAS}"

    retries = 10
    while retries > 0:
        try:
            model = mlflow.pyfunc.load_model(model_uri)
            logger.info(f"Model {MLFLOW_MODEL_NAME} (Alias: {MLFLOW_MODEL_ALIAS}) loaded successfully.")
            break
        except Exception as e:
            logger.warning(f"Failed to load model (MLflow server may not be ready): {e}")
            retries -= 1
            if retries == 0:
                logger.error("Could not load model after several retries. Exiting.")
                raise
            logger.info("Retrying in 5 seconds...")
            await asyncio.sleep(5)

    logger.info("Starting Kafka consumer...")
    kafka_consumer_task = asyncio.create_task(consume_messages())

    yield

    # Shutdown: cancel Kafka consumer
    if kafka_consumer_task:
        kafka_consumer_task.cancel()
        try:
            await kafka_consumer_task
        except asyncio.CancelledError:
            logger.info("Kafka consumer task cancelled.")


# --- FastAPI App ---
app = FastAPI(title="GigaFlow Model Service", lifespan=lifespan)

app.add_middleware(PrometheusMiddleware)
app.add_route("/metrics", metrics)


# --- Health Check Endpoint ---
@app.get("/health")
async def health_check():
    status = {
        "status": "ok",
        "model_loaded": model is not None,
        "kafka_consumer_running": kafka_consumer_task is not None and not kafka_consumer_task.done(),
    }
    # Check DB connectivity
    try:
        async with engine.connect() as conn:
            await conn.execute(sqlalchemy.text("SELECT 1"))
        status["database"] = "connected"
    except Exception:
        status["database"] = "disconnected"
        status["status"] = "degraded"

    if not model:
        status["status"] = "degraded"

    status_code = 200 if status["status"] == "ok" else 503
    if status_code == 503:
        raise HTTPException(status_code=503, detail=status)
    return status


# --- Model Reload Endpoint ---
class ReloadRequest(BaseModel):
    version: int | None = None  # If None, loads champion


@app.post("/reload")
async def reload_model(request: ReloadRequest = ReloadRequest()):
    """Hot-swap the model. Optionally specify a version number."""
    global model
    if request.version:
        model_uri = f"models:/{MLFLOW_MODEL_NAME}/{request.version}"
    else:
        model_uri = f"models:/{MLFLOW_MODEL_NAME}@{MLFLOW_MODEL_ALIAS}"
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info(f"Model reloaded: {model_uri}")
        return {"status": "reloaded", "uri": model_uri}
    except Exception as e:
        logger.error(f"Failed to reload model: {e}")
        raise HTTPException(status_code=500, detail=f"Reload failed: {e}") from e


# --- Prediction Endpoint ---
@app.post("/predict")
async def predict(request: PredictionRequest):
    """Performs a live sentiment prediction on a single text input."""
    global model
    if not model:
        raise HTTPException(status_code=503, detail="Model is not loaded yet. Please wait.")

    try:
        PREDICTION_INPUT_LENGTH.observe(len(request.text))
        data_df = pd.DataFrame({"text": [request.text]})

        start = _time.perf_counter()
        prediction = model.predict(data_df)
        PREDICTION_LATENCY.observe(_time.perf_counter() - start)

        sentiment_score = float(prediction[0])
        sentiment_label = "Positive" if sentiment_score == 1.0 else "Negative"
        PREDICTIONS_TOTAL.labels(sentiment=sentiment_label).inc()

        return {"text": request.text, "sentiment_label": sentiment_label, "sentiment_score": sentiment_score}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}") from e


# --- Kafka Consumer Logic ---
async def consume_messages():
    """
    Asynchronous background task to consume messages from Kafka
    and write predictions to PostgreSQL.
    """
    consumer = None
    delay = 5

    while True:
        try:
            logger.info(f"Attempting to connect consumer to Kafka at {KAFKA_SERVER}...")
            consumer = AIOKafkaConsumer(
                KAFKA_TOPIC,
                bootstrap_servers=KAFKA_SERVER,
                value_deserializer=lambda v: json.loads(v.decode("utf-8")),
                group_id="aiokafka-consumer",
            )
            await consumer.start()
            logger.info("Kafka consumer connected successfully.")
            break
        except Exception as e:
            logger.warning(f"Consumer connection failed: {e}. Retrying in {delay} seconds...")
            await asyncio.sleep(delay)
            delay = min(delay * 2, 60)
            if consumer:
                await consumer.stop()

    logger.info("Kafka consumer started and polling...")

    try:
        async for msg in consumer:
            try:
                kafka_msg = KafkaMessage(**msg.value)
            except Exception as e:
                logger.warning(f"Invalid Kafka message, skipping: {e}")
                continue

            if not model:
                logger.warning("Model not loaded, skipping message.")
                continue

            PREDICTION_INPUT_LENGTH.observe(len(kafka_msg.text))
            data_df = pd.DataFrame({"text": [kafka_msg.text]})

            start = _time.perf_counter()
            prediction = model.predict(data_df)
            PREDICTION_LATENCY.observe(_time.perf_counter() - start)

            sentiment_score = float(prediction[0])
            sentiment_label = "Positive" if sentiment_score == 1.0 else "Negative"
            PREDICTIONS_TOTAL.labels(sentiment=sentiment_label).inc()

            logger.info(f"Prediction: '{kafka_msg.text}' -> {sentiment_label}")

            async with AsyncSessionLocal() as session, session.begin():
                session.add(
                    SentimentPrediction(
                        text=kafka_msg.text,
                        sentiment_score=sentiment_score,
                        sentiment_label=sentiment_label,
                        message_timestamp=datetime.datetime.utcfromtimestamp(kafka_msg.timestamp),
                    )
                )
            logger.info("Prediction saved to database.")

    except Exception as e:
        logger.error(f"Consumer error: {e}")
    finally:
        if consumer:
            await consumer.stop()
            logger.info("Kafka consumer stopped.")
