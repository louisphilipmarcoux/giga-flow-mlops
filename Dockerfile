# Inference Dockerfile — uses pre-built base image with all deps.
# Rebuilds in seconds since only source code changes.
FROM louisphilip/gigaflow-base:inference

# Install additional packages not in base image
USER root
RUN /opt/venv/bin/pip install --no-cache-dir slowapi==0.1.9 redis==5.2.1
USER appuser
WORKDIR /home/appuser/app

COPY --chown=appuser:appuser src/ src/
COPY --chown=appuser:appuser scripts/ scripts/

ENV PATH="/opt/venv/bin:$PATH"
EXPOSE 8000
