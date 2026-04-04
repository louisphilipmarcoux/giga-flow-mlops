# Kubernetes Deployment

Alternative deployment using Kubernetes instead of Docker Compose.

## Prerequisites

- Kubernetes cluster (minikube, kind, or cloud provider)
- kubectl configured
- Base images pushed to Docker Hub

## Quick Start

```bash
# Create namespace
kubectl create namespace gigaflow

# Apply secrets (edit values first!)
kubectl apply -f k8s/secrets.yaml -n gigaflow

# Deploy model service with HPA
kubectl apply -f k8s/model-service.yaml -n gigaflow

# Check status
kubectl get pods -n gigaflow
kubectl get hpa -n gigaflow
```

## Notes

- The model service uses HorizontalPodAutoscaler to scale 1-5 replicas based on CPU usage
- Secrets should be managed with a proper secrets manager (Vault, AWS Secrets Manager) in production
- For full deployment, add manifests for Kafka, PostgreSQL, MinIO, MLflow, etc.
- Consider using Helm charts for more complex deployments
