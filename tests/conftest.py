import pytest

def pytest_addoption(parser):
    parser.addoption(
        "--model-service-url", action="store", default="http://localhost:8000",
        help="Base URL for the model service to test"
    )
    parser.addoption(
        "--mlflow-tracking-uri", action="store", default="http://localhost:5000",
        help="MLflow tracking URI for promotion tests"
    )
    parser.addoption(
        "--minio-endpoint-url", action="store", default="http://localhost:9000",
        help="MinIO endpoint URL"
    )