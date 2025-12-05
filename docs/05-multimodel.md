# Multi-Model Support

The system includes a full **multi-model routing architecture**, designed to resemble production LLM orchestration platforms. Although the v1.0.0 deployment runs in **single-model mode**, the internals are engineered to support multiple models behind one API gateway.

In practice today:

- **Local (non-Docker) deployment** defaults to:  
  meta-llama/Llama-3.1-8B-Instruct
- **Docker deployment** defaults to:  
  meta-llama/Llama-3.2-1B-Instruct

These defaults are controlled via environment variables and container image configuration, not via `models.yaml` (which is currently dormant by default).

---

## Architecture Overview

The backend ships with:

- **MultiModelManager** — a container for many model backends
- **ModelManager** — local PyTorch inference (CPU or MPS)
- **HttpLLMClient** — remote inference via HTTP
- **Model-aware caching** — cache keys include the model ID
- **Declarative configuration** — support for a models.yaml file
- **Per-request model selection** — via the "model" field in `/v1/generate`

This architecture enables:

- Switching models on a per-request basis
- Mixing local and remote models (hybrid inference)
- A/B testing and shadow traffic
- Versioned models (e.g., `foo-7b-v1` → `foo-7b-v2`)
- Future cluster / multi-node deployments

Even though only one model is active per deployment in v1.0.0, the plumbing is built for multi-model workloads.

---

## Current Default Behavior (v1.0.0)

For stability and resource predictability, **v1.0.0 runs in single-model mode**:

- The **local dev flow** (uv run) loads a single Llama 3.1 8B model.
- The **Docker stack** loads a single Llama 3.2 1B model.
- The `models.yaml` file is not used by default; it is intentionally kept as an **opt-in** advanced configuration.

This means:

- You get consistent behavior out of the box.
- Multi-model routing can be turned on later without an API-breaking refactor.
- The project still showcases a realistic production-style architecture under the hood.

---

## Why Ship Multi-Model Architecture Now?

There are strong engineering reasons to include the design, even if it’s not fully activated yet:

### 1. Avoid future breaking changes

Model routing affects:

- Cache keys and cache invalidation
- Database schema and indexing strategy
- Request/response contracts (e.g., "model" field)
- Metrics, logging, and dashboards

Designing these concerns up front lets you evolve the system without disruptive rewrites.

### 2. Portfolio and realism

This project aims to resemble how real AI infrastructure is built. Multi-model routing is a core feature of:

- Hosted LLM providers (OpenAI, Anthropic, etc.)
- Platform offerings (AWS Bedrock, Azure OpenAI)
- Internal inference gateways at large organizations

Having a MultiModelManager and clearly separated local/remote clients demonstrates systems thinking beyond “single model demo.”

### 3. Ready for a post-v1.0.0 hardening cycle

Once v1.0.0 is shipped and stabilized, you can:

- Wire in `models.yaml`
- Load multiple models at startup
- Add a model selector into the UI
- Turn on advanced routing policies

All of that can be done without reworking the core APIs or cache/datastore layout.

---

## Example models.yaml (Disabled by default)

When you’re ready to enable multi-model routing, you can introduce a `models.yaml` similar to:

    default_model: mistralai/Mistral-7B-v0.1

    models:
      - id: mistralai/Mistral-7B-v0.1
        type: local

      - id: deepseek-ai/DeepSeek-R1
        type: remote
        base_url: http://deepseek-node:8000

      - id: microsoft/phi-2
        type: remote
        base_url: http://phi2-node:8000

Then a client could issue:

    POST /v1/generate
    {
      "prompt": "Explain transformers in one sentence.",
      "model": "microsoft/phi-2"
    }

The MultiModelManager would route the request to the appropriate backend (local or remote), and the caching layer would key responses by both the prompt and the model ID.

---

## Future Work After v1.0.0

Once multi-model mode is explicitly enabled, planned enhancements include:

- A model selector in the web UI
- Per-model concurrency limits and warm pools
- Per-model Prometheus metrics and Grafana dashboards
- Tiered caching strategies (fast local vs slower remote models)
- A/B test routing policies (traffic splitting between models)
- Shadow deployments and gradual rollouts for new models

All of these features fit naturally into the existing architecture; they don’t require a redesign.

---

## Summary

- **Today (v1.0.0):**
  - Local: Llama 3.1 8B Instruct
  - Docker: Llama 3.2 1B Instruct
  - Single-model per deployment for simplicity.

- **Architecture:**
  - Multi-model ready (MultiModelManager, local/remote separation, model-aware caching).

- **Roadmap:**
  - Activate `models.yaml` and per-request model selection in a future, hardening-focused release without breaking the current API.

The system is “multi-model capable” even though it deliberately runs **one model per deployment** in v1.0.0.