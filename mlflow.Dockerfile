FROM python:3.11-slim

WORKDIR /app

# 1. Install system dependencies
RUN apt-get update && \
    apt-get install -y dirmngr gnupg build-essential && \
    rm -rf /var/lib/apt/lists/*

# 2. Install Python dependencies
# Using a requirements file is even better, but for a direct fix:
RUN pip install --no-cache-dir mlflow==2.9.2 gunicorn sqlalchemy

# 3. Expose the port
EXPOSE 5000

# 4. Set the default command
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "sqlite:////db/mlflow.db", "--registry-store-uri", "sqlite:////db/mlflow.db", "--artifacts-destination", "file:///mlruns", "--serve-artifacts"]