import asyncio
import json
import mlflow
import pandas as pd
from fastapi import FastAPI
from aiokafka import AIOKafkaConsumer

# --- Configuration (UPDATED) ---
KAFKA_TOPIC = "giga-flow-messages"
KAFKA_SERVER = "localhost:9092"
MLFLOW_MODEL_NAME = "giga-flow-sentiment"
MLFLOW_MODEL_ALIAS = "champion" # Use the alias we just set

# --- FastAPI App ---
app = FastAPI(title="GigaFlow Model Service")

# --- Model Loading ---
model = None

@app.on_event("startup")
async def startup_event():
    """
    On app startup:
    1. Load the 'champion' model from MLflow Registry using its alias.
    2. Start the Kafka consumer background task.
    """
    global model
    print("Loading MLflow model...")
    
    # --- THIS IS THE UPDATED URI ---
    # We now use the format models:/<name>@<alias>
    model_uri = f"models:/{MLFLOW_MODEL_NAME}@{MLFLOW_MODEL_ALIAS}"
    # -------------------------------
    
    model = mlflow.pyfunc.load_model(model_uri)
    
    print(f"Model {MLFLOW_MODEL_NAME} (Alias: {MLFLOW_MODEL_ALIAS}) loaded successfully.")
    
    # Start the Kafka consumer in the background
    print("Starting Kafka consumer...")
    asyncio.create_task(consume_messages())

# --- Health Check Endpoint ---
@app.get("/health")
async def health_check():
    """
    Simple health check to confirm the service is running.
    """
    return {"status": "ok"}

# --- Kafka Consumer Logic ---
async def consume_messages():
    """
    Asynchronous background task to consume messages from Kafka.
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
            # A new message has arrived
            print(f"\nReceived message: {msg.value}")
            
            # Extract the text
            text_input = msg.value.get('text')
            
            if text_input and model:
                # 1. Prepare data for the model (it expects a DataFrame)
                data_df = pd.DataFrame({'text': [text_input]})
                
                # 2. Run inference
                prediction = model.predict(data_df)
                sentiment = 'Positive' if prediction[0] == 1 else 'Negative'
                
                # 3. Process result
                # In Phase 3, we will write this to PostgreSQL.
                # For now, we just print it.
                print(f"Prediction: '{text_input}' -> {sentiment}")
                
    finally:
        # Clean up
        await consumer.stop()
        print("Kafka consumer stopped.")

# To run this app (from the root folder):
# uvicorn src.model_service.main:app --reload