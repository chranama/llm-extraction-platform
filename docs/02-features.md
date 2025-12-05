# Features Overview

LLM Server provides a comprehensive set of features modeled after real production LLM infrastructure.  
This section summarizes each major capability and the engineering rationale behind it.

---

## 🚀 Core API Gateway Features

### 1. API Key Authentication

Every protected endpoint (including `/v1/generate`) requires a valid API key stored in Postgres.

Supports:

- Key activation / deactivation  
- Role-based access (bootstrap vs normal keys, future: admin roles)  
- Quota and rate limits tied to keys  

This mirrors how commercial LLM APIs secure access.

---

### 2. Rate Limiting

Requests are limited per minute based on key role.

Behavior:

- Exceed limit → HTTP 429 with `rate_limit_exceeded`  
- Configurable via environment variables  
- Implemented in a way that can later be swapped out for Redis-backed distributed rate limiting  

This protects the service from accidental or malicious overload.

---

### 3. Usage Quotas

Monthly quota system backed by Postgres.

Features:

- Tracks tokens generated per key  
- Resets monthly (configurable)  
- Enforced on each request  
- Exceed quota → HTTP 402 with `quota_exhausted`  

This simulates billing-style enforcement and supports realistic usage dashboards.

---

### 4. Token Counting

Token accounting is performed for:

- Usage logs  
- Quota enforcement  
- Metrics reporting  

This gives you per-key and per-model usage visibility (a prerequisite for billing, capacity planning, and analytics).

---

## 🧠 Model Inference Features

### 5. Multi-Model Routing (MultiModelManager)

Supports:

- Multiple local models (e.g. different checkpoints)  
- Multiple remote models reachable over HTTP  
- Dynamic model selection via a `model` field in the request  

If no model is specified, the configured default model is used.

This allows you to:

- A/B test models  
- Route specific workloads to specific models  
- Gradually introduce new models into production

---

### 6. Local Execution (CPU or MPS)

The server can run models on:

- CPU (Linux/macOS)  
- Apple Silicon MPS (when `MODEL_DEVICE=mps`)  

Configuration is done via `.env`. The runtime is intentionally simple so you can later add a CUDA-enabled image or connect to a remote GPU.

---

### 7. Remote Execution (HTTP LLM Clients)

Models can be served from other hosts via:

- An `HttpLLMClient` abstraction  
- Per-model `base_url` configuration in `models.yaml`

This enables:

- Multi-node LLM clusters  
- Hybrid setups (some models local, some remote)  
- Integration with other LLM-serving backends

---

## ⚡ Caching & Performance

### 8. Completion Cache (Postgres)

Primary deduplication layer.

Cache key includes:

- Model ID  
- Prompt hash  
- Generation parameters fingerprint  

Benefits:

- Massive speedup on repeated prompts  
- Reduced compute load and token usage  
- Cache survives restarts (backed by the DB)  

The cache is transparent to clients.

---

### 9. Optional Redis Cache (Future High-Throughput Mode)

A second-level, in-memory cache designed for:

- Ultra-low latency responses  
- Handling high QPS workloads  
- Acting as a shared cache in multi-node deployments  

Redis will complement, not replace, the Postgres cache (DB remains the source of truth).

---

## 📊 Observability & DevOps

### 10. Prometheus Metrics

All critical platform metrics are exported under `/metrics`.

Metric categories include:

- Request throughput and latency  
- Error rates by status code  
- Cache hit / miss statistics  
- Token usage  
- Model-level performance  
- Health / readiness signals  

These metrics are ready to be scraped by Prometheus and visualized in Grafana.

---

### 11. Grafana Dashboards

Ready-to-use dashboards (provisioned via Docker Compose) connect to Prometheus and Postgres.

Example insights:

- API latency and p95 / p99 stats  
- Per-model throughput  
- Cache performance  
- Per-key and per-endpoint usage  
- System health over time  

The default Grafana URL in the stack is:

- `http://localhost:8080/grafana/`

---

### 12. Structured Logging

The system records structured logs into Postgres, including:

- `inference_logs` (per-call metadata, durations, token usage, cached vs non-cached)  
- `completion_cache` (cache hits / misses)  
- `api_keys` (including active / inactive state, quotas, labels)

This makes it easy to build internal analytics and audit views.

---

## 🔧 Developer & Deployment Features

### 13. Docker Compose Stack

The provided `docker-compose.yml` spins up:

- FastAPI LLM server (`api`)  
- Postgres (`postgres`)  
- Redis (`redis`)  
- Prometheus (`prometheus`)  
- Grafana (`grafana`)  
- pgAdmin (`pgadmin`)  
- UI playground (`ui`)  
- Nginx front-end (`nginx`)

This gives you a reproducible, production-style environment with a single command (`make up`).

---

### 14. UI Playground

A minimal frontend at:

- `http://localhost:8080/ui/`

Features:

- Prompt input  
- Output display  
- Uses the same `/v1/generate` API as external clients  
- Sends the configured API key via `X-API-Key`  

It is intentionally simple and serves primarily as:

- A smoke-test UI  
- A way to visually inspect latency, caching, and error behavior  

---

### 15. Test Suite (pytest)

The Python test suite covers:

- API key validation  
- Rate limiting behavior  
- Quota enforcement  
- Cache hit / miss paths  
- Model routing logic  
- Error semantics (e.g., `missing_api_key` vs `invalid_api_key`)  
- Health checks (`/healthz`, `/readyz`)  

The v1.0.0 focus is **coverage of critical paths**; deeper hardening and additional scenarios are planned for the post-1.0 “stabilization” phase.