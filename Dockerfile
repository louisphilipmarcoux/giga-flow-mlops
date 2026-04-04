# Inference Dockerfile — uses pre-built base image with all deps.
# Rebuilds in seconds since only source code changes.
#
# Base image includes: FastAPI, PyTorch (CPU), transformers, MLflow,
# Kafka, PostgreSQL, Redis, slowapi, Evidently, and all connectors.
FROM louisphilip/gigaflow-base:inference

USER appuser
WORKDIR /home/appuser/app

COPY --chown=appuser:appuser src/ src/
COPY --chown=appuser:appuser scripts/ scripts/

ENV PATH="/opt/venv/bin:$PATH"
EXPOSE 8000
