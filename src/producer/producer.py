import json
import logging
import os
import random
import time

from kafka import KafkaProducer

logger = logging.getLogger("producer")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

# Dummy data to simulate a live feed
dummy_data = [
    "I love this product, it's amazing!",
    "This is the worst service I have ever received.",
    "The new update is fantastic.",
    "I'm so frustrated with this app.",
    "What a wonderful experience!",
    "It's okay, but I expected more.",
    "Absolutely brilliant, exceeded my expectations!",
    "Terrible quality, would not recommend to anyone.",
    "The customer support was very helpful and responsive.",
    "I regret buying this, total waste of money.",
    "Great value for the price, very satisfied.",
    "The interface is confusing and hard to navigate.",
    "Best purchase I've made this year!",
    "Disappointing performance, expected much better.",
    "Smooth and seamless experience from start to finish.",
]

# Kafka topic to send messages to
TOPIC_NAME = "giga-flow-messages"
KAFKA_SERVER = os.getenv("KAFKA_SERVER", "localhost:9092")


def init_producer():
    """Initialize KafkaProducer with retries and exponential backoff."""
    delay = 5
    while True:
        try:
            logger.info(f"Attempting to connect to Kafka at {KAFKA_SERVER}...")
            producer = KafkaProducer(
                bootstrap_servers=KAFKA_SERVER,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                compression_type="lz4",
            )
            logger.info("Kafka producer connected successfully.")
            return producer
        except Exception as e:
            logger.warning(f"Connection failed: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
            delay = min(delay * 2, 60)


# Initialize Kafka Producer with retries
producer = init_producer()


def send_message():
    """Simulates sending a single message to the Kafka topic."""
    message = {"text": random.choice(dummy_data), "timestamp": time.time()}

    logger.info(f"Sending message: {message}")
    producer.send(TOPIC_NAME, value=message)
    producer.flush()


if __name__ == "__main__":
    logger.info("Starting data producer...")
    logger.info(f"Sending messages to Kafka topic: '{TOPIC_NAME}'")

    while True:
        send_message()
        time.sleep(random.randint(1, 5))
