# Training Dockerfile — uses pre-built base image with training deps.
# Rebuilds in seconds since only source code changes.
FROM louisphilip/gigaflow-base:training

USER appuser
WORKDIR /home/appuser/app

COPY --chown=appuser:appuser src/ src/
COPY --chown=appuser:appuser scripts/ scripts/

ENV PATH="/opt/venv/bin:$PATH"
EXPOSE 8000
