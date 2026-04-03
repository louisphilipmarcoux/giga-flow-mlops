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
    # Positive
    "I love this product, it's amazing!",
    "The new update is fantastic.",
    "What a wonderful experience!",
    "Absolutely brilliant, exceeded my expectations!",
    "The customer support was very helpful and responsive.",
    "Great value for the price, very satisfied.",
    "Best purchase I've made this year!",
    "Smooth and seamless experience from start to finish.",
    # Negative
    "This is the worst service I have ever received.",
    "I'm so frustrated with this app.",
    "Terrible quality, would not recommend to anyone.",
    "I regret buying this, total waste of money.",
    "The interface is confusing and hard to navigate.",
    "Disappointing performance, expected much better.",
    # Neutral
    "It's okay, but I expected more.",
    "The product arrived on time and works as described.",
    "I have no strong feelings about this one way or another.",
    "It does what it says, nothing more nothing less.",
    # Curious / Confused
    "I wonder how this compares to the competition?",
    "Can someone explain how this feature works?",
    "I'm not sure what to think about the new design.",
    # Surprise
    "I can't believe how fast the shipping was!",
    "Wow, I did not expect this level of quality!",
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
