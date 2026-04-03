.PHONY: up down build test lint train logs clean help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

up: ## Start all services
	docker compose up -d --build

down: ## Stop all services
	docker compose down

build: ## Build all Docker images
	docker compose build

test: ## Run pytest suite (requires services running)
	MLFLOW_TRACKING_URI=http://localhost:5000 python -m pytest tests/ -v \
		--model-service-url=http://localhost:8000 \
		--mlflow-tracking-uri=http://localhost:5000 \
		--minio-endpoint-url=http://localhost:9000

test-unit: ## Run unit tests only (no services needed)
	MLFLOW_TRACKING_URI=http://localhost:5000 python -m pytest tests/test_kafka_validation.py tests/test_producer.py tests/test_promotion.py -v \
		--model-service-url=http://localhost:8000 \
		--mlflow-tracking-uri=http://localhost:5000 \
		--minio-endpoint-url=http://localhost:9000

lint: ## Run ruff linter and formatter
	ruff check src/ scripts/ tests/
	ruff format --check src/ scripts/ tests/

lint-fix: ## Fix linting issues automatically
	ruff check --fix src/ scripts/ tests/
	ruff format src/ scripts/ tests/

train: ## Run model training inside producer container
	docker compose exec -e MLFLOW_TRACKING_URI=http://mlflow_server:5000 producer \
		/bin/bash -c "jupyter nbconvert --to script src/notebooks/01_model_training.ipynb && python -u src/notebooks/01_model_training.py"

promote: ## Run model promotion script
	docker compose exec -e MLFLOW_TRACKING_URI=http://mlflow_server:5000 producer \
		python scripts/promote_model.py

logs: ## Tail logs from all services
	docker compose logs -f

logs-model: ## Tail model service logs
	docker compose logs -f model_service

logs-producer: ## Tail producer logs
	docker compose logs -f producer

status: ## Show service health status
	docker compose ps

clean: ## Stop services and remove volumes
	docker compose down --volumes --remove-orphans

dvc-pull: ## Pull data from DVC remote
	dvc pull

install: ## Install development dependencies
	pip install -r requirements-dev.txt
	pre-commit install
