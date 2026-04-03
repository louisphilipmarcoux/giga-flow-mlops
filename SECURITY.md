# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| main    | Yes                |

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

1. **Do not** open a public GitHub issue
2. Email the maintainer directly or use [GitHub's private vulnerability reporting](../../security/advisories/new)
3. Include a description of the vulnerability, steps to reproduce, and potential impact
4. Allow up to 72 hours for an initial response

## Security Considerations

This project is designed as an MLOps reference implementation. When deploying to production:

- Change all default credentials in `.env`
- Enable TLS for inter-service communication
- Restrict network access to management interfaces (MLflow, Grafana, MinIO Console)
- Use a secrets manager for credential storage
- Enable authentication on all exposed services
- Review container images for known vulnerabilities
