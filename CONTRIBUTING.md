# Contributing to GigaFlow MLOps

Thank you for your interest in contributing! This guide will help you get started.

## Development Setup

### Prerequisites

- Docker & Docker Compose
- Python 3.11+
- Git

### Local Development

```bash
# Clone the repository
git clone https://github.com/your-username/giga-flow-mlops.git
cd giga-flow-mlops

# Create environment file
cp .env.example .env

# Install dev dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Start all services
make up

# Run tests
make test
```

## Code Standards

- **Linting**: We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting
- **Pre-commit hooks**: Run automatically on every commit (ruff, trailing whitespace, YAML/JSON validation)
- **Type hints**: Use type annotations for function signatures
- **Logging**: Use the `logging` module, never `print()`
- **Tests**: Add tests for new functionality

## Pull Request Process

1. Fork the repository and create a feature branch from `main`
2. Make your changes following the code standards above
3. Add or update tests as appropriate
4. Ensure all tests pass: `make test`
5. Ensure linting passes: `make lint`
6. Update documentation if needed
7. Submit a pull request using the PR template

## Commit Messages

Use clear, descriptive commit messages:

```
fix: resolve Kafka consumer reconnection issue
feat: add model rollback endpoint
docs: update monitoring setup guide
test: add drift monitor integration tests
```

## Reporting Issues

- Use the [Bug Report](../../issues/new?template=bug_report.yml) template for bugs
- Use the [Feature Request](../../issues/new?template=feature_request.yml) template for enhancements
- Check existing issues before creating a new one

## Architecture

See the [README](README.md) for the architecture overview. Key components:

| Component | Location | Technology |
|-----------|----------|------------|
| Inference API | `src/model_service/` | FastAPI |
| Data Producer | `src/producer/` | kafka-python |
| Drift Monitor | `src/drift_monitor/` | Evidently AI |
| Dashboard | `src/dashboard/` | Streamlit |
| Training | `src/notebooks/` | Jupyter + MLflow |
| Promotion | `scripts/` | MLflow Client |
| Tests | `tests/` | pytest |

## Questions?

Open a [Discussion](../../discussions) or file an issue.
