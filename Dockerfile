# We will use a multi-stage build for efficiency
# --- STAGE 1: Build the 'dependencies' image ---
FROM python:3.11-slim AS builder

# Set the working directory
WORKDIR /app

# Install build-essential tools for some Python packages
RUN apt-get update && apt-get install -y build-essential

# Copy only the requirements file first to leverage Docker caching
COPY requirements.txt .

# Create a virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip
# Cache-buster v1
RUN pip install --no-cache-dir -r requirements.txt

# --- STAGE 2: Build the final 'runtime' image ---
FROM python:3.11-slim

# Set a non-root user for security
RUN useradd -m appuser
USER appuser
WORKDIR /home/appuser/app

# Copy the virtual environment from the 'builder' stage
COPY --chown=appuser:appuser --from=builder /opt/venv /opt/venv

# Copy the application source code
COPY --chown=appuser:appuser src/ src/

# Set the PATH to use the venv
ENV PATH="/opt/venv/bin:$PATH"

# Expose the port the app runs on
EXPOSE 8000

# This command will be run by our docker-compose.yml
# We don't need a CMD or ENTRYPOINT here, as compose will handle it.