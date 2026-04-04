# Base image with all training dependencies pre-installed.
# Build once, push to Docker Hub, and inherit in training Dockerfile.
#
# Build & push:
#   docker build -f base-training.Dockerfile -t louisphilip/gigaflow-base:training .
#   docker push louisphilip/gigaflow-base:training
FROM python:3.14-slim

RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl && \
    rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements-inference.txt .
COPY requirements-training.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-training.txt && \
    rm requirements-inference.txt requirements-training.txt

RUN useradd -m appuser
