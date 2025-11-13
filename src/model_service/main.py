import asyncio
import json
import pandas as pd
import datetime
import os
from fastapi import FastAPI
from aiokafka import AIOKafkaConsumer
import sqlalchemy
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, Integer, String, Float, DateTime
from starlette_prometheus import PrometheusMiddleware, metrics

# --- Configuration ---
KAFKA_TOPIC = "giga-flow-messages"
KAFKA_SERVER = os.getenv("KAFKA_SERVER", "localhost:9092")
MLFLOW_MODEL_NAME = "giga-flow-sentiment"
MLFLOW_MODEL_ALIAS = "champion"

# Load from environment variable, with our local DB as a fallback
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://gigaflow:password@localhost:5432/sentiment_db")

import mlflow

# --- SQLAlchemy Setup ---
engine = create_async_engine(DATABASE_URL, echo=True)
Base = declarative_base()
AsyncSessionLocal = sessionmaker(
    bind=engine, class_=AsyncSession, expire_on_commit=False
)

# --- Define Predictions Table (ORM Model) ---
class SentimentPrediction(Base):
    __tablename__ = "sentiment_predictions"
    id = Column(Integer, primary_key=True, index=True)
    text = Column(String, index=True)
    sentiment_score = Column(Float) # Use Float for scores, or Integer for 0/1
    sentiment_label = Column(String)
    message_timestamp = Column(DateTime)
    processed_at = Column(DateTime, default=datetime.datetime.utcnow)

# --- FastAPI App ---
app = FastAPI(title="GigaFlow Model Service")

app.add_middleware(PrometheusMiddleware)

# --- Model & DB Globals ---
model = None
db_engine = None

async def create_db_and_tables():
    """Create the table in the database."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

@app.on_event("startup")
async def startup_event():
    """
    On app startup:
    1. Create database tables.
    2. Load the MLflow model (with retries).
    3. Start the Kafka consumer.
    """
    global model, engine
    
    print("Creating database tables...")
    await create_db_and_tables()
    print("Database tables created.")

    print("Loading MLflow model...")
    model_uri = f"models:/{MLFLOW_MODEL_NAME}@{MLFLOW_MODEL_ALIAS}"
    
    # --- NEW RETRY LOOP FOR MLFLOW ---
    retries = 10
    while retries > 0:
        try:
            model = mlflow.pyfunc.load_model(model_uri)
            print(f"Model {MLFLOW_MODEL_NAME} (Alias: {MLFLOW_MODEL_ALIAS}) loaded successfully.")
            break  # Success!
        except Exception as e:
            print(f"Failed to load model (MLflow server may not be ready): {e}")
            retries -= 1
            if retries == 0:
                print("Could not load model after several retries. Exiting.")
                raise  # Raise the last exception and crash the service
            print("Retrying in 5 seconds...")
            await asyncio.sleep(5)
    # --- END OF RETRY LOOP ---

    print("Starting Kafka consumer...")
    asyncio.create_task(consume_messages())

# --- Health Check Endpoint ---
@app.get("/health")
async def health_check():
    return {"status": "ok"}

# --- Kafka Consumer Logic (UPDATED) ---
async def consume_messages():
    """
    Asynchronous background task to consume messages from Kafka
    and write predictions to PostgreSQL.
    """
    consumer = None
    
    while True:
        try:
            print(f"Attempting to connect consumer to Kafka at {KAFKA_SERVER}...")
            consumer = AIOKafkaConsumer(
                KAFKA_TOPIC,
                bootstrap_servers=KAFKA_SERVER,
                value_deserializer=lambda v: json.loads(v.decode('utf-8'))
            )
            # Start the consumer. This is the part that will fail if Kafka isn't ready.
            await consumer.start()
            print("Kafka consumer connected successfully.")
            break  # Exit the loop if connection is successful
        except Exception as e:
            print(f"Consumer connection failed: {e}. Retrying in 5 seconds...")
            await asyncio.sleep(5)
            if consumer:
                # Ensure we stop the partially started consumer before retrying
                await consumer.stop()

    # If we're here, the consumer started successfully
    print("Kafka consumer started and polling...")
    
    try:
        async for msg in consumer:
            print(f"\nReceived message: {msg.value}")
            text_input = msg.value.get('text')
            msg_timestamp = msg.value.get('timestamp')
            
            if text_input and model:
                data_df = pd.DataFrame({'text': [text_input]})
                prediction = model.predict(data_df)
                
                # Convert prediction to friendly format
                sentiment_score = float(prediction[0]) # Assuming model returns 0 or 1
                sentiment_label = 'Positive' if sentiment_score == 1.0 else 'Negative'
                
                print(f"Prediction: '{text_input}' -> {sentiment_label}")

                # --- Write to Database ---
                async with AsyncSessionLocal() as session:
                    async with session.begin():
                        prediction_record = SentimentPrediction(
                            text=text_input,
                            sentiment_score=sentiment_score,
                            sentiment_label=sentiment_label,
                            message_timestamp=datetime.datetime.fromtimestamp(msg_timestamp)
                        )
                        session.add(prediction_record)
                        await session.commit()
                print("Prediction saved to database.")
                
    finally:
        if consumer:
            await consumer.stop()
            print("Kafka consumer stopped.")