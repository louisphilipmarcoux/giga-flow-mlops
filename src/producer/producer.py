import time
import json
import random
import os
from kafka import KafkaProducer

# Dummy data to simulate a live feed
dummy_data = [
    "I love this product, it's amazing!",
    "This is the worst service I have ever received.",
    "The new update is fantastic.",
    "I'm so frustrated with this app.",
    "What a wonderful experience!",
    "It's okay, but I expected more."
]

# Kafka topic to send messages to
TOPIC_NAME = "giga-flow-messages"
KAFKA_SERVER = os.getenv("KAFKA_SERVER", "localhost:9092")

# --- producer.py (Modified init_producer function) ---

def init_producer():
    """Initialize KafkaProducer with retries to handle startup race."""
    # CHANGED: Use an infinite loop to wait for Kafka indefinitely
    while True: 
        try:
            print(f"Attempting to connect to Kafka at {KAFKA_SERVER}...")
            producer = KafkaProducer(
                bootstrap_servers=KAFKA_SERVER,
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
            print("Kafka producer connected successfully.")
            return producer
        except Exception as e:
            print(f"Connection failed: {e}. Retrying in 5 seconds...")
            # Removed the retry countdown and exit logic
            time.sleep(5)

# Initialize Kafka Producer with retries
producer = init_producer()

def send_message():
    """
    Simulates sending a single message to the Kafka topic.
    """
    message = {
        'text': random.choice(dummy_data),
        'timestamp': time.time()
    }
    
    print(f"Sending message: {message}")
    
    # Send the message
    producer.send(TOPIC_NAME, value=message)
    producer.flush() # Ensure all messages are sent

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    print("Starting data producer...")
    print(f"Sending messages to Kafka topic: '{TOPIC_NAME}'")
    
    # Run an infinite loop to simulate a continuous stream
    while True:
        send_message()
        # Wait for a random time (1-5 seconds)
        time.sleep(random.randint(1, 5))