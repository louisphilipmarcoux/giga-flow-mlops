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
from sqlalchemy import Column, DateTime, Float, Integer, String, Text
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
EMOTIONS_TOTAL = Counter("emotions_total", "Total emotions detected", ["emotion"])

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
    top_emotion = Column(String, nullable=True)
    emotions_json = Column(Text, nullable=True)
    message_timestamp = Column(DateTime)
    processed_at = Column(DateTime, default=datetime.datetime.utcnow)


# --- Emotion-to-Sentiment Mapping ---
POSITIVE_EMOTIONS = {
    "admiration",
    "amusement",
    "approval",
    "caring",
    "desire",
    "excitement",
    "gratitude",
    "joy",
    "love",
    "optimism",
    "pride",
    "relief",
}
NEGATIVE_EMOTIONS = {
    "anger",
    "annoyance",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "fear",
    "grief",
    "nervousness",
    "remorse",
    "sadness",
}
NEUTRAL_EMOTIONS = {"neutral", "realization", "confusion", "curiosity"}


def emotions_to_sentiment(top_emotion):
    """Derive sentiment from the top detected emotion."""
    if top_emotion in POSITIVE_EMOTIONS:
        return "Positive"
    if top_emotion in NEGATIVE_EMOTIONS:
        return "Negative"
    return "Neutral"


def parse_prediction(prediction_row):
    """Parse a single prediction row from the model output.
    Handles both old binary format (0/1) and new emotion format (JSON string)."""
    val = prediction_row
    if isinstance(val, str):
        try:
            data = json.loads(val)
            if isinstance(data, dict):
                top_emotion = data.get("top_emotion", "")
                emotions = data.get("emotions", {})

                # Override to Neutral if top emotion is neutral-category
                if top_emotion in NEUTRAL_EMOTIONS:
                    data["sentiment_label"] = "Neutral"
                    data["sentiment_score"] = 0.5
                # Mixed signal: weak positive emotion + negative emotions present → Neutral
                elif top_emotion in POSITIVE_EMOTIONS:
                    top_score = emotions.get(top_emotion, 1.0)
                    neg_total = sum(emotions.get(e, 0) for e in emotions if e in NEGATIVE_EMOTIONS)
                    if top_score < 0.5 or (neg_total > top_score * 0.3):
                        data["sentiment_label"] = "Neutral"
                        data["sentiment_score"] = 0.5

                return data
        except (json.JSONDecodeError, TypeError):
            pass

    # Old binary model fallback
    score = float(val)
    sentiment = "Positive" if score == 1.0 else "Negative"
    return {
        "sentiment_label": sentiment,
        "sentiment_score": score,
        "top_emotion": sentiment.lower(),
        "emotions": {},
    }


# --- Model & Consumer Globals ---
model = None
model_info = {"uri": None, "version": None, "alias": None}
kafka_consumer_task = None


async def create_db_and_tables():
    """Create the table in the database."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


@asynccontextmanager
async def lifespan(app):
    """Manage startup and shutdown lifecycle."""
    global model, model_info, kafka_consumer_task

    logger.info("Creating database tables...")
    await create_db_and_tables()
    logger.info("Database tables created.")

    logger.info("Loading MLflow model...")
    model_uri = f"models:/{MLFLOW_MODEL_NAME}@{MLFLOW_MODEL_ALIAS}"

    retries = 10
    while retries > 0:
        try:
            model = mlflow.pyfunc.load_model(model_uri)
            model_info = {"uri": model_uri, "version": "champion", "alias": MLFLOW_MODEL_ALIAS}
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


@app.get("/health")
async def health_check():
    status = {
        "status": "ok",
        "model_loaded": model is not None,
        "kafka_consumer_running": kafka_consumer_task is not None and not kafka_consumer_task.done(),
    }
    try:
        async with engine.connect() as conn:
            await conn.execute(sqlalchemy.text("SELECT 1"))
        status["database"] = "connected"
    except Exception:
        status["database"] = "disconnected"
        status["status"] = "degraded"

    if not model:
        status["status"] = "degraded"

    if status["status"] != "ok":
        raise HTTPException(status_code=503, detail=status)
    return status


@app.get("/model")
async def get_model_info():
    """Returns info about the currently loaded model."""
    return model_info


class ReloadRequest(BaseModel):
    version: int | None = None


@app.post("/reload")
async def reload_model(request: ReloadRequest | None = None):
    if request is None:
        request = ReloadRequest()
    """Hot-swap the model. Optionally specify a version number."""
    global model, model_info
    if request.version:
        model_uri = f"models:/{MLFLOW_MODEL_NAME}/{request.version}"
        label = f"v{request.version}"
    else:
        model_uri = f"models:/{MLFLOW_MODEL_NAME}@{MLFLOW_MODEL_ALIAS}"
        label = "champion"
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        model_info = {"uri": model_uri, "version": label, "alias": MLFLOW_MODEL_ALIAS if not request.version else None}
        logger.info(f"Model reloaded: {model_uri}")
        return {"status": "reloaded", **model_info}
    except Exception as e:
        logger.error(f"Failed to reload model: {e}")
        raise HTTPException(status_code=500, detail=f"Reload failed: {e}") from e


@app.post("/predict")
async def predict(request: PredictionRequest):
    """Performs sentiment + emotion prediction on a single text input."""
    global model
    if not model:
        raise HTTPException(status_code=503, detail="Model is not loaded yet. Please wait.")

    try:
        PREDICTION_INPUT_LENGTH.observe(len(request.text))
        data_df = pd.DataFrame({"text": [request.text]})

        start = _time.perf_counter()
        prediction = model.predict(data_df)
        PREDICTION_LATENCY.observe(_time.perf_counter() - start)

        result = parse_prediction(prediction.iloc[0])
        sentiment_label = result["sentiment_label"]
        PREDICTIONS_TOTAL.labels(sentiment=sentiment_label).inc()

        top_emotion = result.get("top_emotion", "")
        if top_emotion:
            EMOTIONS_TOTAL.labels(emotion=top_emotion).inc()

        return {
            "text": request.text,
            "sentiment_label": sentiment_label,
            "sentiment_score": result.get("sentiment_score", 0),
            "top_emotion": top_emotion,
            "emotions": result.get("emotions", {}),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}") from e


# --- Kafka Consumer Logic ---
async def consume_messages():
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

            result = parse_prediction(prediction.iloc[0])
            sentiment_label = result["sentiment_label"]
            top_emotion = result.get("top_emotion", "")
            emotions = result.get("emotions", {})
            PREDICTIONS_TOTAL.labels(sentiment=sentiment_label).inc()
            if top_emotion:
                EMOTIONS_TOTAL.labels(emotion=top_emotion).inc()

            logger.info(f"'{kafka_msg.text}' -> {sentiment_label} | {top_emotion}")

            async with AsyncSessionLocal() as session, session.begin():
                session.add(
                    SentimentPrediction(
                        text=kafka_msg.text,
                        sentiment_score=result.get("sentiment_score", 0),
                        sentiment_label=sentiment_label,
                        top_emotion=top_emotion,
                        emotions_json=json.dumps(emotions) if emotions else None,
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
