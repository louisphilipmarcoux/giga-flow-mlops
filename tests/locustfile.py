"""
Load testing for GigaFlow Model Service.

Usage:
    pip install locust
    locust -f tests/locustfile.py --host http://localhost:8000

Then open http://localhost:8089 to configure and start the test.
"""

import random

from locust import HttpUser, between, task

SAMPLE_TEXTS = [
    "I absolutely love this product!",
    "This is the worst service I've ever received.",
    "It's okay, nothing special.",
    "J'adore ce produit, c'est fantastique!",
    "Das ist wirklich schrecklich.",
    "¡Me encanta este producto!",
    "I'm curious about how this works.",
    "Terrible quality, would not recommend.",
    "The product arrived on time and works fine.",
    "Wow, this completely blew my mind!",
    "I can't believe how bad this is.",
    "Average product, meets basic expectations.",
    "The customer support was very helpful.",
    "I regret buying this, total waste of money.",
    "What a pleasant surprise, much better than expected!",
]


class ModelServiceUser(HttpUser):
    """Simulates a user making prediction requests."""

    wait_time = between(0.5, 2)

    @task(10)
    def predict(self):
        """Send a prediction request."""
        self.client.post(
            "/predict",
            json={"text": random.choice(SAMPLE_TEXTS)},
        )

    @task(3)
    def health_check(self):
        """Check service health."""
        self.client.get("/health")

    @task(1)
    def model_info(self):
        """Check loaded model info."""
        self.client.get("/model")

    @task(1)
    def feedback_stats(self):
        """Check feedback statistics."""
        self.client.get("/feedback/stats")

    @task(2)
    def submit_feedback(self):
        """Submit feedback on a prediction."""
        text = random.choice(SAMPLE_TEXTS)
        self.client.post(
            "/feedback",
            json={
                "original_text": text,
                "original_sentiment": random.choice(["Positive", "Negative", "Neutral"]),
                "original_emotion": random.choice(["joy", "anger", "neutral", "surprise"]),
                "is_correct": random.random() > 0.2,
            },
        )
