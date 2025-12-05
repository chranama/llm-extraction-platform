# Testing Guide

LLM Server ships with a structured test suite covering authentication, quotas, rate limits, generation, caching, health checks, and integration behavior.  
This section describes how tests are organized and how to run them.

---

## 1. Test Layout

The test suite lives in the `tests/` directory:

    tests/
      ├── conftest.py
      ├── test_auth.py
      ├── test_auth_rate_quota.py
      ├── test_generate.py
      ├── test_health.py
      ├── test_health_metrics.py
      ├── test_integrate_generate.py
      └── test_limits.py

Each file targets a specific subsystem:

| File                          | Focus Area                                  |
|------------------------------|---------------------------------------------|
| `test_auth.py`               | API key validation, missing/invalid keys    |
| `test_auth_rate_quota.py`    | rate limiting and quota consumption logic   |
| `test_generate.py`           | core text-generation behavior               |
| `test_integrate_generate.py` | end-to-end flow through cache + generation  |
| `test_health.py`             | `/healthz` and `/readyz` semantics          |
| `test_health_metrics.py`     | `/metrics` endpoint and metric shape        |
| `test_limits.py`             | low-level rate/ quota helpers               |

`conftest.py` provides shared fixtures, including:

- an app test client
- a temporary DB with migrations applied
- seeded API keys
- mocked LLM calls (fast, deterministic)

---

## 2. Running Tests (Local / Host Mode)

This is the recommended way to develop and harden the system.

1. Install dependencies (once):

    uv sync

2. Run the full suite:

    uv run pytest -q

### Running Specific Tests

Single file:

    uv run pytest tests/test_generate.py

Single test:

    uv run pytest tests/test_generate.py::test_basic_generation

Verbose output:

    uv run pytest -vv

---

## 3. Running Against the Dockerized API (Optional)

For smoke-testing the containerized deployment:

1. Start the stack:

    make up

2. Point tests at the running API:

    API_BASE=http://localhost:8000 uv run pytest -q

Notes:

- Some tests rely on Python-level monkeypatching and are primarily designed for in-process (local) mode.
- Container mode is best for “does this deployment behave correctly?” checks, not day-to-day TDD.

---

## 4. What the Tests Guarantee

The current suite validates several critical behaviors:

### Authentication

- Missing `X-API-Key` → `401`, code `missing_api_key`
- Invalid or inactive key → `401`, code `invalid_api_key`

### Rate Limiting

- Requests beyond a key’s RPM threshold return `429`, code `rate_limit_exceeded`
- Windows reset correctly and allow traffic again

### Quotas

- Generations consume quota for that key
- When quota is exceeded, responses return `402`, code `quota_exhausted`
- Monthly reset timestamps are respected

### Caching

- Identical `(model, prompt, params)` combinations hit the completion cache
- Different parameters bypass the cache
- Cached responses are returned with `cached=true` in the payload

### Health & Readiness

- `/healthz` returns `200` when the process is alive
- `/readyz` verifies DB/Redis/model readiness and fails fast when dependencies break

### Metrics

- Core HTTP counters and histograms exist on `/metrics`
- LLM-specific metrics (latency, request counts) are exported with reasonable labels

### Generation Semantics

- `/v1/generate` returns a stable JSON schema: `{model, output, cached}`
- Parameters such as `max_new_tokens`, `temperature`, `top_p`, `top_k`, `stop` are validated
- Error paths map to the documented status codes and error payloads

---

## 5. Testing Philosophy

The suite is intentionally:

- **Deterministic** – LLM calls are mocked, so tests are not subject to model randomness.
- **Fast** – a full run should complete in seconds, not minutes.
- **Subsystem-oriented** – each concern (auth, limits, health, etc.) is isolated.
- **Production-flavored** – it mirrors patterns seen in real infra stacks.

For the post-1.0 hardening phase, likely additions include:

- more scenario-level tests (multi-step flows)
- optional tests for multimodel routing when `models.yaml` is enabled
- baseline performance and regression tests

---

## 6. CI / Automation (Future Work)

If you later add CI (for example GitHub Actions), a minimal job would:

    uv sync
    uv run pytest -q

You can extend this to a matrix of:

- Python versions
- local vs container mode
- Redis enabled vs disabled

---

## 7. When to Add New Tests

You should add or update tests when:

- introducing new endpoints or changing response schemas
- modifying authentication, quotas, or rate limits
- changing caching strategy or adding Redis-backed behavior
- enabling / extending multimodel routing
- touching cross-cutting concerns like logging or metrics

Keeping tests aligned with new behavior is key to preserving the API guarantees documented in the “API Stability and Versioning” section.