## API Stability and Versioning

This project exposes a small, opinionated API surface. Not all endpoints are equally stable.

This section defines:

- Which endpoints are considered **stable** for the 1.x series
- Which endpoints are **experimental / internal**
- What kinds of changes you can expect over time

The goal is to make it clear what you can safely integrate with in other projects.

---

### Versioning Policy

The API is versioned under the `/v1/...` prefix.

For the **1.x** line:

- **Stable endpoints**
  - Path and HTTP method are guaranteed not to change without a breaking version bump (e.g., `/v2/...`).
  - Core request/response fields will remain compatible:
    - New optional fields may be added.
    - Existing fields will not change semantics or types in a breaking way.
- **Experimental endpoints**
  - May change shape, behavior, or be removed between minor releases.
  - Intended for internal use, operator workflows, and future features.

The project uses a **“backend-for-self” first** philosophy: stable endpoints should be safe for external use; experimental ones are allowed to evolve quickly.

---

### Stable Endpoints (1.x Contract)

These endpoints are treated as **stable** and safe to build against for the entire 1.x series.

| Path                 | Method | Description                                       |
|----------------------|--------|---------------------------------------------------|
| `/healthz`           | GET    | Liveness probe (process is up).                   |
| `/readyz`            | GET    | Readiness probe (DB + LLM + Redis checks).        |
| `/metrics`           | GET    | Prometheus metrics in text exposition format.     |
| `/v1/generate`       | POST   | Core non-streaming text generation endpoint.      |
| `/v1/models`         | GET    | Lists default model and available model IDs.      |
| `/v1/me/usage`       | GET    | Usage summary for the current API key.            |

#### Stability notes

- **`/v1/generate`**
  - Stable request fields:
    - `prompt` (string)
    - `max_new_tokens` (int, optional)
    - `temperature` (float, optional)
    - `top_p` (float, optional)
    - `top_k` (int, optional)
    - `stop` (array of strings, optional)
    - `model` (string, optional — interpreted against the active deployment configuration)
  - Stable response fields:
    - `model` (string; the resolved model_id actually used)
    - `output` (string; generated text)
    - `cached` (bool; whether the completion came from cache)

- **`/v1/models`**
  - Stable response structure:
    - `default_model` (string)
    - `models` (array of strings, available model IDs)

- **`/v1/me/usage`**
  - Stable behavior:
    - Authenticated with `X-API-Key`.
    - Returns a JSON summary of usage associated with that key.
  - Structure may grow with additional optional fields (e.g. more granular time windows), but existing fields will remain compatible.

- **`/metrics`**
  - Exposes standard Prometheus metrics.
  - Individual metric *names/labels* may grow over time, but existing ones will not change meaning arbitrarily.

The set of stable endpoints may grow in future **minor** releases (e.g. 1.1.x), but not shrink.

---

### Experimental / Internal Endpoints

These endpoints are provided for convenience and operational use, but are considered **experimental** in the 1.x series. They may change more aggressively.

| Path                         | Method | Status        | Description                                         |
|------------------------------|--------|---------------|-----------------------------------------------------|
| `/v1/generate/batch`         | POST   | Experimental  | Batch generation over multiple prompts.             |
| `/v1/stream`                 | POST   | Experimental  | SSE-based streaming generation.                     |
| `/v1/admin/keys`             | GET/POST | Experimental | Admin key management (list/create).                 |
| `/v1/admin/logs`             | GET    | Experimental  | Query inference logs.                               |

**Why experimental?**

- `/v1/generate/batch`  
  - Interface and behavior may change as batching, caching, and logging strategies evolve.
  - Token accounting and response shape might be refined (e.g. totals, per-item metadata).

- `/v1/stream`  
  - Streaming format is SSE-based today but could evolve (chunk structure, metadata, end-of-stream markers).

- `/v1/admin/*`  
  - These are primarily **operator** endpoints.
  - Filter parameters, pagination behavior, and payload shapes are likely to evolve as operational needs become clearer.

If you use these experimental endpoints in downstream projects, you should expect breaking changes across minor versions (e.g., 1.1 → 1.2).

---

### Non-API (Reverse-Proxy) Entry Points

The Docker / Nginx stack also exposes convenience URLs for UI and observability. These are **not** part of the stable API surface, but are documented for completeness:

- UI playground (proxied): `http://localhost:8080/ui/`
- Prometheus via Nginx: `http://localhost:8080/prometheus/`
- Grafana via Nginx: `http://localhost:8080/grafana/`
- pgAdmin via Nginx: `http://localhost:8080/pgadmin/`

Underlying “direct” ports (e.g. `http://localhost:9090` for Prometheus, `:3000` for Grafana, `:5050` for pgAdmin) are considered **deployment details**, not part of the public API contract. They may change between versions or across deployment modes.

---

### How to Use This in Practice

- If you’re building **client libraries, other services, or automation**, rely on **Stable** endpoints only.
- If you’re experimenting with new features (batching, streaming, admin ops), you can use **Experimental** endpoints, but:
  - Pin to a specific version (e.g. `1.0.0`).
  - Expect to revisit your integration when you upgrade.

Future 2.x would be the place for any true breaking changes to the stable surface (e.g. a redesigned `/v2/generate`).