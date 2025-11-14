import pytest
import httpx
import os
import time

# Get the service URL from an env var, or default to localhost
# This allows us to run tests against different environments
MODEL_SERVICE_URL = os.getenv("MODEL_SERVICE_URL", "http://localhost:8001") # We'll map it to 8001 in compose

# Define the client as a fixture for reuse
@pytest.fixture(scope="module")
def model_service_url(request):
    """Fixture to get the --model-service-url from command line."""
    return request.config.getoption("--model-service-url")

@pytest.fixture(scope="module")
def client(model_service_url):
    # Wait for the service to be healthy
    retries = 10
    while retries > 0:
        try:
            response = httpx.get(f"{model_service_url}/health")
            if response.status_code == 200:
                print(f"Model service at {model_service_url} is healthy.")
                break
        except httpx.RequestError:
            print(f"Waiting for model service at {model_service_url}...")
            time.sleep(5)
            retries -= 1
    if retries == 0:
        pytest.fail(f"Could not connect to model service at {model_service_url}.")
        
    with httpx.Client(base_url=model_service_url) as client:
        yield client

def test_health_check(client):
    """
    Test the /health endpoint.
    """
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict_endpoint(client):
    """
    Test the /predict endpoint with valid input.
    """
    payload = {"text": "I love this MLOps pipeline!"}
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 200
    
    data = response.json()
    assert data["text"] == payload["text"]
    assert "sentiment_label" in data
    assert "sentiment_score" in data
    assert data["sentiment_label"] in ["Positive", "Negative"]
    assert isinstance(data["sentiment_score"], float)

def test_predict_endpoint_negative(client):
    """
    Test the /predict endpoint with negative text.
    """
    payload = {"text": "This is a terrible product."}
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment_label"] == "Negative"

def test_predict_endpoint_empty_text(client):
    """
    Test the /predict endpoint with a request that is missing the 'text' field.
    FastAPI should return a 422 Unprocessable Entity.
    """
    payload = {"foo": "bar"} # Missing 'text'
    response = client.post("/predict", json=payload)
    assert response.status_code == 422