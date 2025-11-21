# LLM Server  
### A Production-Style API Gateway + Inference Runtime for Large Language Models

This project is a self-hosted, production-inspired **LLM serving platform** built with FastAPI, Hugging Face Transformers, and PyTorch. It is designed to mirror real-world architecture patterns used by companies deploying foundation models behind internal and external APIs.

Rather than being just a demo model runner, this repository focuses on:

- Infrastructure
- Security
- Quotas
- Observability
- Scalability patterns
- Clean separation of concerns

It is designed to be a **portfolio-grade system-level project** demonstrating my ability to design and build ML/AI infrastructure.

---

## Key Features

- FastAPI-based LLM gateway
- API key authentication
- Rate limiting + concurrency limits
- Usage quotas (monthly limits)
- Prometheus metrics + Grafana dashboards
- SQLAlchemy + Alembic for persistence
- Streaming & non-streaming generation
- Hugging Face Transformers backend
- Docker support
- Test suite with Pytest

The architecture cleanly separates:

- Gateway API — validates requests, manages users/quotas, logs history
- LLM Runtime Service — loads and serves the actual model
- Database layer — tracks users, API keys, quotas and request history
- Metrics layer — exposes structured telemetry

This design follows real patterns used in production AI platforms.

---

## Architecture Overview

Client  
→ FastAPI Gateway (Auth · Quotas · Rate Limit · Logging · Metrics)  
→ LLM Runtime API (HuggingFace + PyTorch)  
→ GPU / CPU Inference Engine  

Supporting components:

- Database (Postgres / SQLite) for users, keys, quotas
- Prometheus for metrics collection
- Grafana for visualization
- Docker / Compose for deployment

This is intentionally separated from the typical “single-file chatbot” pattern and instead follows:

- Microservice thinking
- Separation of concerns
- Horizontal scalability concepts

---

## Project Structure

    llm-server/
    ├── app/
    │   ├── gateway/              # FastAPI gateway (auth, quota, limits)
    │   ├── runtime/              # Model loading & inference engine
    │   ├── db/                   # SQLAlchemy models + Alembic migrations
    │   ├── core/                 # Config, security, settings
    │   ├── metrics/              # Prometheus metrics endpoints
    │   └── main.py               # Main FastAPI entrypoint
    │
    ├── models/                   # Local models (optional)
    ├── tests/                    # Pytest test suite
    ├── docker/                   # Docker + compose configs
    ├── scripts/                  # Admin & bootstrap scripts
    ├── requirements.txt
    ├── alembic.ini
    ├── docker-compose.yml
    └── README.md

---

## Core Capabilities

### 1. API Gateway Features

- API key validation
- User tiers / plans
- Monthly quotas
- Per-minute rate limiting
- Request logging
- Request tracing
- Abuse & replay protection

This makes the project closer to:

**“OpenAI-style backend for your own models”**

than a traditional ML demo.

---

### 2. Inference Layer

The runtime supports:

- Streaming token generation
- Standard completion
- Chat-style input
- Model configuration (temperature, max tokens, etc.)
- GPU optimization (if available)

Designed to support models such as:

- Mistral
- LLaMA
- Phi
- DeepSeek
- Custom HuggingFace models

---

### 3. Observability

Integrated metrics include:

- Total requests
- Tokens generated
- Latency (p95 / p99 ready)
- Errors by type
- Active clients
- Quota usage

These feed into:

- Prometheus
- Grafana dashboards

This is a rare and valuable feature in portfolio projects.

---

## Testing

Tests are written using Pytest and focus on:

- API key validation
- Quota enforcement
- Rate limiting
- Request contracts
- Edge cases

To run tests:

    pytest

This project intentionally includes tests because production systems are not notebooks.

---