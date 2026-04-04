# Base image with all ML/inference dependencies pre-installed.
# Build once, push to Docker Hub, and inherit in app Dockerfiles.
#
# Build & push:
#   docker build -f base.Dockerfile -t louisphilip/gigaflow-base:inference .
#   docker push louisphilip/gigaflow-base:inference
FROM python:3.14-slim

RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl && \
    rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements-inference.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-inference.txt && \
    rm requirements-inference.txt

RUN useradd -m appuser
