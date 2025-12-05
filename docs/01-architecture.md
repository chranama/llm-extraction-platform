# Architecture Overview

LLM Server follows a production-inspired architecture that mirrors how modern AI infrastructure teams design secure, observable, scalable LLM inference systems. The platform is organized into five major layers:

1. **API Gateway**
2. **Model Orchestration**
3. **Inference Runtimes (local + remote)**
4. **Caching & Persistence**
5. **Observability Stack**

Below is the architecture diagram in pure ASCII format, fully compatible with Markdown renderers:

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

## Layer Descriptions

### 1. API Gateway (FastAPI)
The entrypoint for all model requests. Responsibilities include:
- Authentication via API Keys  
- Rate limiting and usage quotas  
- Token counting  
- Logging and request metadata  
- Routing to selected model (`model` field in request)  
- Exposing `/metrics`, `/healthz`, and `/readyz`

### 2. MultiModelManager
A routing layer that allows multiple models (local or remote) to coexist.

It supports:
- Local execution via Python ModelManager  
- Remote execution via HttpLLMClient  
- Dynamic model selection per request  

Configured via `models.yaml`.

### 3. Inference Runtimes
LLM Server runs inference on:

- **CPU** (default)
- **Apple Silicon MPS**
- **Remote GPU hosts** (via HTTP)
- **Future GPU / multi-node runtimes**

This layer abstracts execution device selection, model loading, batching, and token streaming.

### 4. Persistence & Caching
The system provides two integrated caching layers:

- **Postgres CompletionCache** (primary)
- **Optional Redis Hot Cache** for high-throughput deployments

Postgres also stores API keys, usage logs, and inference logs.

### 5. Observability
The stack includes:

- **Prometheus** for request, rate, latency, and error metrics  
- **Grafana** dashboards  
- **Structured logs** (inference logs, request logs, usage logs)  
- **Health endpoints** for readiness/liveness probes  

This mirrors real-world LLM platform monitoring.