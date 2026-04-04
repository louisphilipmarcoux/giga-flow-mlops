import asyncio
import datetime
import hashlib
import json
import logging
import os
import random
import time as _time
from contextlib import asynccontextmanager

import mlflow
import pandas as pd
import redis
import sqlalchemy
from aiokafka import AIOKafkaConsumer
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.security import APIKeyHeader
from prometheus_client import Counter, Histogram
from pydantic import BaseModel, field_validator
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
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
LANGUAGES_TOTAL = Counter("languages_total", "Languages detected", ["language"])
TOXIC_TOTAL = Counter("toxic_predictions_total", "Toxic predictions detected")

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


class FeedbackRequest(BaseModel):
    prediction_id: int | None = None
    original_text: str
    original_sentiment: str
    original_emotion: str | None = None
    corrected_sentiment: str | None = None
    corrected_emotion: str | None = None
    is_correct: bool


# --- Configuration ---
KAFKA_TOPIC = "giga-flow-messages"
KAFKA_SERVER = os.getenv("KAFKA_SERVER", "localhost:9092")
MLFLOW_MODEL_NAME = "giga-flow-sentiment"
MLFLOW_MODEL_ALIAS = "champion"

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://gigaflow:password@localhost:5432/sentiment_db")

# Auth & Rate Limiting
API_KEYS = set(os.getenv("API_KEYS", "").split(",")) - {""}  # comma-separated keys, empty = no auth
RATE_LIMIT = os.getenv("RATE_LIMIT", "60/minute")
AUTH_ENABLED = bool(API_KEYS)

# Redis Cache
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))  # 1 hour default

# Auxiliary model names (loaded separately from MLflow sentiment model)
LANG_MODEL_NAME = os.getenv("LANG_MODEL_NAME", "papluca/xlm-roberta-base-language-detection")
TOXIC_MODEL_NAME = os.getenv("TOXIC_MODEL_NAME", "unitary/toxic-bert")
ENABLE_MULTITASK = os.getenv("ENABLE_MULTITASK", "true").lower() == "true"

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
    language = Column(String, nullable=True)
    toxicity_score = Column(Float, nullable=True)
    is_toxic = Column(sqlalchemy.Boolean, nullable=True)
    message_timestamp = Column(DateTime)
    processed_at = Column(DateTime, default=datetime.datetime.utcnow)


class PredictionFeedback(Base):
    __tablename__ = "prediction_feedback"
    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, nullable=True)
    original_text = Column(String)
    original_sentiment = Column(String)
    original_emotion = Column(String, nullable=True)
    corrected_sentiment = Column(String, nullable=True)
    corrected_emotion = Column(String, nullable=True)
    is_correct = Column(sqlalchemy.Boolean)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


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
lang_pipeline = None
toxic_pipeline = None

# --- A/B Testing ---
challenger_model = None
challenger_info = {"uri": None, "version": None}
AB_SPLIT = float(os.getenv("AB_SPLIT", "0.0"))  # 0.0 = no A/B, 0.2 = 20% challenger
AB_PREDICTIONS = Counter("ab_predictions_total", "A/B test predictions", ["variant"])


def load_auxiliary_models():
    """Load language detection and toxicity models."""
    global lang_pipeline, toxic_pipeline
    if not ENABLE_MULTITASK:
        logger.info("Multi-task detection disabled.")
        return

    from transformers import pipeline as hf_pipeline

    try:
        logger.info(f"Loading language detection model: {LANG_MODEL_NAME}")
        lang_pipeline = hf_pipeline("text-classification", model=LANG_MODEL_NAME, truncation=True, max_length=512)
        logger.info("Language detection model loaded.")
    except Exception as e:
        logger.warning(f"Failed to load language model: {e}")

    try:
        logger.info(f"Loading toxicity model: {TOXIC_MODEL_NAME}")
        toxic_pipeline = hf_pipeline("text-classification", model=TOXIC_MODEL_NAME, truncation=True, max_length=512)
        logger.info("Toxicity model loaded.")
    except Exception as e:
        logger.warning(f"Failed to load toxicity model: {e}")


def detect_language(text):
    """Detect language of text. Returns language code (e.g., 'en', 'fr')."""
    if lang_pipeline is None:
        return None
    try:
        result = lang_pipeline(text[:512])
        return result[0]["label"]
    except Exception:
        return None


def detect_toxicity(text):
    """Detect toxicity of text. Returns (score, is_toxic)."""
    if toxic_pipeline is None:
        return None, None
    try:
        result = toxic_pipeline(text[:512])
        label = result[0]["label"]
        score = result[0]["score"]
        is_toxic = label.lower() == "toxic" and score > 0.5
        return round(score, 4), is_toxic
    except Exception:
        return None, None


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

    load_auxiliary_models()

    logger.info("Starting Kafka consumer...")
    kafka_consumer_task = asyncio.create_task(consume_messages())

    yield

    if kafka_consumer_task:
        kafka_consumer_task.cancel()
        try:
            await kafka_consumer_task
        except asyncio.CancelledError:
            logger.info("Kafka consumer task cancelled.")


# --- Rate Limiter ---
limiter = Limiter(key_func=get_remote_address, default_limits=[RATE_LIMIT])

# --- Redis Cache ---
redis_client = None
try:
    redis_client = redis.from_url(REDIS_URL, decode_responses=True, socket_connect_timeout=2)
    redis_client.ping()
    logger.info("Redis cache connected.")
except Exception:
    redis_client = None
    logger.info("Redis not available — caching disabled.")


def cache_get(text: str) -> dict | None:
    """Get cached prediction for text."""
    if redis_client is None:
        return None
    try:
        key = f"pred:{hashlib.md5(text.encode()).hexdigest()}"
        cached = redis_client.get(key)
        if cached:
            return json.loads(cached)
    except Exception:
        pass
    return None


def cache_set(text: str, result: dict):
    """Cache a prediction result."""
    if redis_client is None:
        return
    try:
        key = f"pred:{hashlib.md5(text.encode()).hexdigest()}"
        redis_client.setex(key, CACHE_TTL, json.dumps(result))
    except Exception:
        pass


# --- API Key Authentication ---
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str | None = Depends(api_key_header)):
    """Verify API key if authentication is enabled."""
    if not AUTH_ENABLED:
        return True
    if api_key is None or api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return True


# --- FastAPI App ---
app = FastAPI(title="GigaFlow Model Service", lifespan=lifespan)
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return HTTPException(status_code=429, detail="Rate limit exceeded. Try again later.")


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


class ABTestRequest(BaseModel):
    challenger_version: int
    split: float = 0.2  # fraction of traffic to challenger


@app.post("/ab-test")
async def setup_ab_test(request: ABTestRequest):
    """Set up A/B testing between champion and a challenger version."""
    global challenger_model, challenger_info, AB_SPLIT
    model_uri = f"models:/{MLFLOW_MODEL_NAME}/{request.challenger_version}"
    try:
        challenger_model = mlflow.pyfunc.load_model(model_uri)
        challenger_info = {"uri": model_uri, "version": f"v{request.challenger_version}"}
        AB_SPLIT = max(0.0, min(1.0, request.split))
        logger.info(f"A/B test started: {AB_SPLIT:.0%} traffic to challenger v{request.challenger_version}")
        return {"status": "ab_test_started", "split": AB_SPLIT, **challenger_info}
    except Exception as e:
        logger.error(f"Failed to load challenger: {e}")
        raise HTTPException(status_code=500, detail=f"Challenger load failed: {e}") from e


@app.delete("/ab-test")
async def stop_ab_test():
    """Stop A/B testing, serve only champion."""
    global challenger_model, challenger_info, AB_SPLIT
    challenger_model = None
    challenger_info = {"uri": None, "version": None}
    AB_SPLIT = 0.0
    logger.info("A/B test stopped.")
    return {"status": "ab_test_stopped"}


@app.get("/ab-test")
async def ab_test_status():
    """Get current A/B test status."""
    return {
        "active": challenger_model is not None,
        "split": AB_SPLIT,
        "champion": model_info,
        "challenger": challenger_info,
    }


@app.post("/predict")
@limiter.limit(RATE_LIMIT)
async def predict(request: PredictionRequest, req: Request, _auth: bool = Depends(verify_api_key)):
    """Performs sentiment + emotion prediction with caching, auth, and rate limiting."""
    global model
    if not model:
        raise HTTPException(status_code=503, detail="Model is not loaded yet. Please wait.")

    # Check cache
    cached = cache_get(request.text)
    if cached:
        cached["cached"] = True
        return cached

    try:
        PREDICTION_INPUT_LENGTH.observe(len(request.text))
        data_df = pd.DataFrame({"text": [request.text]})

        # A/B routing
        use_challenger = challenger_model is not None and random.random() < AB_SPLIT
        active_model = challenger_model if use_challenger else model
        variant = "challenger" if use_challenger else "champion"
        AB_PREDICTIONS.labels(variant=variant).inc()

        start = _time.perf_counter()
        prediction = active_model.predict(data_df)
        PREDICTION_LATENCY.observe(_time.perf_counter() - start)

        result = parse_prediction(prediction.iloc[0])
        sentiment_label = result["sentiment_label"]
        PREDICTIONS_TOTAL.labels(sentiment=sentiment_label).inc()

        top_emotion = result.get("top_emotion", "")
        if top_emotion:
            EMOTIONS_TOTAL.labels(emotion=top_emotion).inc()

        # Multi-task: language + toxicity
        language = detect_language(request.text)
        toxicity_score, is_toxic = detect_toxicity(request.text)

        if language:
            LANGUAGES_TOTAL.labels(language=language).inc()
        if is_toxic:
            TOXIC_TOTAL.inc()

        response = {
            "text": request.text,
            "sentiment_label": sentiment_label,
            "sentiment_score": result.get("sentiment_score", 0),
            "top_emotion": top_emotion,
            "emotions": result.get("emotions", {}),
            "language": language,
            "toxicity_score": toxicity_score,
            "is_toxic": is_toxic,
            "model_variant": variant,
            "cached": False,
        }

        # Cache the result
        cache_set(request.text, response)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}") from e


# --- Batch Prediction Endpoint ---
class BatchPredictionRequest(BaseModel):
    texts: list[str]


@app.post("/predict/batch")
@limiter.limit("10/minute")
async def predict_batch(request: BatchPredictionRequest, req: Request, _auth: bool = Depends(verify_api_key)):
    """Batch prediction for multiple texts at once. Max 100 texts per request."""
    global model
    if not model:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    if len(request.texts) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 texts per batch.")

    results = []
    for text in request.texts:
        # Check cache first
        cached = cache_get(text)
        if cached:
            cached["cached"] = True
            results.append(cached)
            continue

        data_df = pd.DataFrame({"text": [text]})
        prediction = model.predict(data_df)
        result = parse_prediction(prediction.iloc[0])
        language = detect_language(text)
        toxicity_score, is_toxic = detect_toxicity(text)

        pred_result = {
            "text": text,
            "sentiment_label": result["sentiment_label"],
            "sentiment_score": result.get("sentiment_score", 0),
            "top_emotion": result.get("top_emotion", ""),
            "language": language,
            "toxicity_score": toxicity_score,
            "is_toxic": is_toxic,
            "cached": False,
        }
        cache_set(text, pred_result)
        results.append(pred_result)

    return {"predictions": results, "count": len(results)}


# --- Explainability Endpoint ---
def _get_positive_score(pred_result):
    """Extract a continuous positive score from prediction result for explainability."""
    emotions = pred_result.get("emotions", {})
    if emotions:
        pos_score = sum(emotions.get(e, 0) for e in POSITIVE_EMOTIONS if e in emotions)
        neg_score = sum(emotions.get(e, 0) for e in NEGATIVE_EMOTIONS if e in emotions)
        return pos_score - neg_score
    return pred_result.get("sentiment_score", 0.5)


@app.post("/explain")
async def explain_prediction(request: PredictionRequest):
    """Explain a prediction by showing word-level contributions.
    Uses leave-one-out perturbation with emotion scores for continuous gradients."""
    global model
    if not model:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    try:
        text = request.text
        words = text.split()
        if len(words) < 2:
            return {"text": text, "explanation": [{"word": text, "importance": 1.0}]}

        # Get baseline score using emotion-derived continuous score
        base_df = pd.DataFrame({"text": [text]})
        base_pred = parse_prediction(model.predict(base_df).iloc[0])
        base_score = _get_positive_score(base_pred)

        # Perturb each word and measure impact
        explanations = []
        for i, word in enumerate(words[:20]):  # Limit for speed
            perturbed = " ".join(words[:i] + words[i + 1 :])
            if not perturbed.strip():
                continue
            perturbed_df = pd.DataFrame({"text": [perturbed]})
            perturbed_pred = parse_prediction(model.predict(perturbed_df).iloc[0])
            perturbed_score = _get_positive_score(perturbed_pred)
            importance = round(base_score - perturbed_score, 4)
            explanations.append({"word": word, "importance": importance})

        return {
            "text": text,
            "sentiment": base_pred["sentiment_label"],
            "explanation": explanations,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explain failed: {str(e)}") from e


# --- Feedback Endpoints ---
@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """Submit feedback on a prediction (correct/incorrect + corrections)."""
    async with AsyncSessionLocal() as session, session.begin():
        session.add(
            PredictionFeedback(
                prediction_id=request.prediction_id,
                original_text=request.original_text,
                original_sentiment=request.original_sentiment,
                original_emotion=request.original_emotion,
                corrected_sentiment=request.corrected_sentiment,
                corrected_emotion=request.corrected_emotion,
                is_correct=request.is_correct,
            )
        )
    logger.info(f"Feedback received: correct={request.is_correct}, text='{request.original_text[:50]}'")
    return {"status": "saved"}


@app.get("/feedback/stats")
async def feedback_stats():
    """Get feedback statistics."""
    async with AsyncSessionLocal() as session:
        total = await session.execute(sqlalchemy.text("SELECT COUNT(*) FROM prediction_feedback"))
        correct = await session.execute(
            sqlalchemy.text("SELECT COUNT(*) FROM prediction_feedback WHERE is_correct = true")
        )
        incorrect = await session.execute(
            sqlalchemy.text("SELECT COUNT(*) FROM prediction_feedback WHERE is_correct = false")
        )
        total_val = total.scalar() or 0
        correct_val = correct.scalar() or 0
        incorrect_val = incorrect.scalar() or 0
        accuracy = correct_val / total_val if total_val > 0 else 0
        return {
            "total_feedback": total_val,
            "correct": correct_val,
            "incorrect": incorrect_val,
            "user_accuracy": round(accuracy, 4),
        }


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

            # Multi-task detection
            language = detect_language(kafka_msg.text)
            toxicity_score, is_toxic = detect_toxicity(kafka_msg.text)
            if language:
                LANGUAGES_TOTAL.labels(language=language).inc()
            if is_toxic:
                TOXIC_TOTAL.inc()

            logger.info(f"'{kafka_msg.text[:60]}' -> {sentiment_label} | {top_emotion} | {language}")

            async with AsyncSessionLocal() as session, session.begin():
                session.add(
                    SentimentPrediction(
                        text=kafka_msg.text,
                        sentiment_score=result.get("sentiment_score", 0),
                        sentiment_label=sentiment_label,
                        top_emotion=top_emotion,
                        emotions_json=json.dumps(emotions) if emotions else None,
                        language=language,
                        toxicity_score=toxicity_score,
                        is_toxic=is_toxic,
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
