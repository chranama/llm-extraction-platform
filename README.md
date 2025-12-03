# LLM Server  
### A Production-Style LLM API Gateway & Inference Runtime

This project is a self-hosted, production-inspired LLM serving platform built with FastAPI, Hugging Face Transformers, and PyTorch.

It mirrors real-world architecture for deploying language models behind authenticated, observable, scalable APIs.

This is a portfolio-grade systems engineering project.

---

## Architecture Diagram (ASCII, safe for all renderers)

Below is a pure-text diagram that renders safely everywhere:

    +--------------------------------------------------------------+
    |                          Client                              |
    |                (CLI / Frontend / Service)                    |
    +--------------+----------------------------+------------------+
                   |                            |
                   v                            |
    +--------------------------------------------------------------+
    |                        FastAPI Gateway                       |
    |--------------------------------------------------------------|
    |  Auth (API Keys)  |  Rate Limits  |  Quotas  |  Logging      |
    |  Metrics (/metrics) | Completion Cache | Routing (/v1/generate) |
    +---------------------------+------------------+----------------+
                                |
                                v
    +--------------------------------------------------------------+
    |                     MultiModelManager                        |
    |  - Local ModelManager                                        |
    |  - Remote HttpLLMClient                                      |
    +---------------------------+------------------+----------------+
                                |
       +------------------------+------------------------+
       |                        |                        |
       v                        v                        v
    +---------+          +-------------+          +--------------+
    |   MPS   |          |     CPU     |          |  Remote GPU  |
    +---------+          +-------------+          +--------------+

    +--------------------------------------------------------------+
    |                       Observability & Data                   |
    |--------------------------------------------------------------|
    | Postgres (Logs + Cache)  |  Prometheus  |  Grafana  | Redis |
    +--------------------------------------------------------------+

---

## Key Features

- FastAPI LLM API Gateway  
- Multi-model routing (MultiModelManager)  
- API key authentication  
- Usage quotas & monthly usage tracking  
- Rate limiting & concurrency limits  
- Token counting  
- Completion cache (DB; Redis optional)  
- Prometheus metrics + Grafana dashboards  
- Evaluate models (GSM8K, MMLU, MBPP, summarization, toxicity)  
- Docker Compose stack  
- uv-based development environment  

---

## Observability URLs

After running:

    make up

These URLs become available:

**API Server**  
    http://localhost:8000  
    http://localhost:8000/metrics  
    http://localhost:8000/healthz  
    http://localhost:8000/readyz  
    http://localhost:8000/v1/models  

**Prometheus**  
    http://localhost:9090  

**Grafana**  
    http://localhost:3000  

**pgAdmin**  
    http://localhost:5050  

---

## Caching Overview

The system implements two caching layers:

### 1. Database Cache (current default)

Each response is cached in the CompletionCache table.

Cache key =  
(model_id, prompt_hash, params_fingerprint)

This provides:

- Deduplication of repeated prompts  
- Faster responses  
- Lower compute load  
- Lower token usage  

### 2. Redis Cache (optional / WIP)

To enable:

    REDIS_ENABLED=true
    REDIS_URL=redis://llm_redis:6379/0

Redis will serve as:

- an in-memory hot cache  
- ultra-low-latency dedupe layer  
- future foundation for distributed batching  

---

## Multi-Model Support

Support for multiple simultaneous models configured via models.yaml.

Example:

    default_model: mistralai/Mistral-7B-v0.1

    models:
      - id: mistralai/Mistral-7B-v0.1
        type: local

      - id: deepseek-ai/DeepSeek-R1
        type: remote
        base_url: http://other-server:8000

      - id: microsoft/phi-2
        type: remote
        base_url: http://phi-node:8000

If no file is provided, the system uses single-model mode.

### Dynamic model selection

    {
        "prompt": "Explain transformers in one sentence",
        "model": "microsoft/phi-2"
    }

List models:

    GET /v1/models

---

## Project Structure

    llm-server/
    ├── src/llm_server/
    │   ├── api/
    │   ├── core/
    │   ├── db/
    │   ├── eval/
    │   ├── services/
    │   └── main.py
    ├── migrations/
    ├── scripts/
    ├── tests/
    ├── models.yaml
    ├── docker-compose.yml
    └── README.md

---

## Makefile Commands

### Lifecycle

    make up
        Start the full stack (API + DB + Redis + Prometheus + Grafana)

    make down
        Stop and remove containers/networks/volumes

    make restart
        Restart only the API containers

    make logs
        Tail logs from all containers

### Database

    make migrate
        Autogenerate a migration

    make upgrade
        Apply migrations

    make downgrade
        Roll back last migration

    make db-shell
        Enter Postgres psql shell

### API Keys

    make seed-key API_KEY=<value>
        Seed an API key into the DB

    make list-keys
        Show all API keys

### Utilities

    make clean
        Remove Python build artifacts

    make nuke
        Destroy all containers and volumes (full reset)

---

## Quickstart

### Quickstart (Containerized)

1. Copy `.env.example` to `.env`
2. Run:

        make up

3. Seed an API key:

        make seed-key API_KEY=<yourkey>

4. Call API:

        curl -X POST http://localhost:8000/v1/generate \
          -H "Content-Type: application/json" \
          -H "X-API-Key: <yourkey>" \
          -d '{"prompt": "Hello!", "max_new_tokens": 32}'

5. Check observability:

        Grafana      → http://localhost:3000
        Prometheus   → http://localhost:9090
        Metrics      → http://localhost:8000/metrics

---

### Quickstart (Local LLM, CPU or MPS)

1. Install environment:

        uv sync --extra cpu

2. Launch the server:

        uv run serve

3. To use Apple Silicon MPS:

        export MODEL_DEVICE=mps

4. Test:

        curl -X POST http://localhost:8000/v1/generate \
          -H "Content-Type: application/json" \
          -H "X-API-Key: <yourkey>" \
          -d '{"prompt": "ping"}'

---

## Admin & Ops APIs

    /v1/me/usage
    /v1/admin/keys
    /v1/admin/logs

Admin keys have elevated permissions.

---

## Testing

Run the test suite:

    uv run pytest

Covers:

- key validation  
- quotas  
- caching  
- model routing  
- streaming  
- rate limiting  
- health checks  

---

## Notes

This repository demonstrates a fully integrated LLM backend platform:

- Gateway  
- Routing  
- Caching  
- Logging  
- Metrics  
- Observability  
- Multi-model orchestration  

It is designed to resemble production systems used by AI infrastructure teams.