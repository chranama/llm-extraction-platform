# LLM Server  
### A production-style API gateway and inference runtime for large language models

This project is a self-hosted, production-inspired LLM serving platform built with FastAPI, Hugging Face Transformers, PyTorch, and SQLAlchemy.  
It mirrors the architecture patterns used by teams that deploy foundation models behind internal and external APIs.

This is not a notebook demo. It is an infrastructure project focused on:

- Clean architecture
- Security and API design
- Quotas and rate limiting
- Observability and telemetry
- Testability and robustness
- Deployment-readiness

It is intended to be a portfolio-grade systems project demonstrating real ML platform engineering skills.

---

## Key Features

- FastAPI-based LLM gateway
- API key authentication
- Per-key quotas and rate limiting
- Request logging and history
- Completion caching
- Prometheus metrics middleware
- Streaming and non-streaming generation
- Local Hugging Face and PyTorch inference
- Extensible provider layer (local / remote / hybrid)
- SQLAlchemy and Alembic for persistence
- Docker support
- Full Pytest test suite
- uv-based dependency and environment management

Architecture layers are cleanly separated:

- API Layer — validation, auth, quotas, logging, metrics
- LLM Layer — model loading and inference execution
- Database Layer — API keys, roles, quotas, request history
- Observability Layer — telemetry and metrics

This closely mirrors real systems used in production AI platforms.

---

## Architecture Overview

Client  
→ FastAPI Gateway (Auth, Quotas, Rate Limit, Logging, Metrics)  
→ Model Manager (Transformers + PyTorch)  
→ GPU / MPS / CPU Inference Engine  

Supporting components:

- Database (Postgres / SQLite via SQLAlchemy)
- Prometheus for metrics
- Redis (optional – caching / queueing)
- Docker and Compose for deployment

The design follows:

- Modular service boundaries
- Dependency injection
- Clear separation of concerns
- Scalability-ready layout

It is intentionally not a single-file chatbot app.

---

## Current Project Structure

    llm-server/
    ├── src/
    │   └── llm_server/
    │       ├── api/                # /generate, /stream, /health
    │       ├── core/               # settings, limits, logging, metrics
    │       ├── db/                 # SQLAlchemy models + session
    │       ├── providers/          # (future) remote/hybrid LLM clients
    │       ├── services/           # ModelManager (HuggingFace, local)
    │       ├── main.py             # FastAPI + lifespan
    │       └── cli.py              # uv run serve
    │
    ├── scripts/                    # Admin / DB scripts
    ├── tests/                      # Full Pytest suite
    ├── migrations/                 # Alembic
    ├── data/                       # Local SQLite (dev/test)
    ├── Dockerfile.api
    ├── pyproject.toml
    ├── uv.lock
    └── README.md

The old app directory has been fully removed and replaced by the modern, scalable src/llm_server layout.

---

## Endpoints

- GET /health — basic liveness
- GET /readyz — model readiness
- POST /v1/generate — text completion
- POST /v1/stream — streaming completion
- GET /metrics — Prometheus metrics

Example request

POST /v1/generate  
Header: X-API-Key: your_api_key  
Body (application/json):

    {
      "prompt": "Explain transformers in one sentence.",
      "max_new_tokens": 50
    }

Example response:

    {
      "model": "your-model-id",
      "output": "...",
      "cached": false
    }

---

## Running locally

### Install

Choose ONE backend:

    uv sync --extra mps      (Apple Silicon)
    uv sync --extra cpu      (CPU)
    uv sync --extra cuda121  (NVIDIA CUDA 12.1)

### Run the server

    uv run serve

The server will start on:

    http://127.0.0.1:8000

---

## Testing

Tests are fully integrated with the FastAPI lifespan and use a dummy LLM to avoid real model downloads.

    uv run pytest

The test suite covers:

- API key validation
- Quota logic
- Rate and concurrency limits
- Generate and stream endpoints
- Health and metrics
- Cache and request logging

This confirms the system behaves like a real production service.

---

## Model System

Models are managed by:

    src/llm_server/services/llm.py

The ModelManager:

- Lazily loads the model
- Supports generate() and stream()
- Works on MPS, CPU, and GPU
- Can be replaced with:
  - Remote API clients
  - Multi-model routing
  - Load-balanced replicas

The provider abstraction lives in:

    src/llm_server/providers/

This allows you to extend the system into:

- Hybrid deployments
- Multi-model switching
- RAG pipelines
- Central model hubs

---

## Observability

- Prometheus metrics exposed at /metrics
- Structured logging per request
- Latency tracking
- Cache hit tracking
- Ready for Grafana dashboards

This is rarely present in portfolio projects and mirrors real platform standards.

---

## Roadmap (Revised)

### Phase 0 – Stability and architecture (completed)

- src-based layout
- Lifespan setup
- Dummy LLM in tests
- All tests passing
- Local ModelManager

### Phase 1 – Deployment hardening

- Finalize Dockerfile.api
- Environment profiles (dev / test / prod)
- Startup validations
- CI pipeline

### Phase 2 – Performance and scalability

- Global concurrency control
- Per-model queueing
- Redis-backed job system
- Multi-model selection

### Phase 3 – Observability and productization

- Dashboards
- Admin UI
- Model management API
- Documentation site

---

This project is intended to grow into:

A self-hosted, open-source LLM platform backend that mirrors the design patterns of OpenAI and Anthropic, but runs under your control.