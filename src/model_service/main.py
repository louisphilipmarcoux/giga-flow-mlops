import asyncio
import json
import mlflow
import pandas as pd
import datetime
from fastapi import FastAPI
from aiokafka import AIOKafkaConsumer

# --- NEW DB IMPORTS ---
import sqlalchemy
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, Integer, String, Float, DateTime

# --- Configuration ---
KAFKA_TOPIC = "giga-flow-messages"
KAFKA_SERVER = "localhost:9092"
MLFLOW_MODEL_NAME = "giga-flow-sentiment"
MLFLOW_MODEL_ALIAS = "champion"

# --- NEW DATABASE CONFIG ---
DATABASE_URL = "postgresql+asyncpg://gigaflow:password@localhost:5432/sentiment_db"

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
    2. Load the MLflow model.
    3. Start the Kafka consumer.
    """
    global model, engine
    
    print("Creating database tables...")
    await create_db_and_tables()
    print("Database tables created.")

    print("Loading MLflow model...")
    model_uri = f"models:/{MLFLOW_MODEL_NAME}@{MLFLOW_MODEL_ALIAS}"
    model = mlflow.pyfunc.load_model(model_uri)
    print(f"Model {MLFLOW_MODEL_NAME} (Alias: {MLFLOW_MODEL_ALIAS}) loaded successfully.")
    
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
    consumer = AIOKafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=KAFKA_SERVER,
        value_deserializer=lambda v: json.loads(v.decode('utf-8'))
    )
    
    await consumer.start()
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

                # --- NEW: Write to Database ---
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
        await consumer.stop()
        print("Kafka consumer stopped.")