## Caching

The system uses a **deterministic, model-aware cache** for completion responses, with
Postgres as the primary backend and Redis available as an optional future extension.

The main goal is to avoid recomputing the same completion when the model and all
generation parameters are identical, while still keeping behavior transparent and
easy to reason about.

### 1. Completion Cache (Postgres)

The primary cache is the `CompletionCache` table in Postgres. Each entry stores:

- `model_id` – the logical model identifier (e.g. `meta-llama/Llama-3.1-8B-Instruct`)
- `prompt` – full prompt text
- `prompt_hash` – SHA-256 hash of the prompt, truncated for storage
- `params_fingerprint` – hash of the generation parameters
- `output` – completion text

The cache key is effectively:

- `(model_id, prompt_hash, params_fingerprint)`

Where:

- `prompt_hash` = hash of the prompt string
- `params_fingerprint` = hash of all non-`prompt`, non-`model` generation parameters:
  - `max_new_tokens`
  - `temperature`
  - `top_p`
  - `top_k`
  - `stop`
  - (plus any future fields added to the request model)

If any of these change, the fingerprint changes, and a new cache entry is created.

#### 1.1 Single request: `/v1/generate`

For `POST /v1/generate`:

1. Resolve `model_id` (single or multi-model configuration).
2. Compute:
   - `prompt_hash` from the incoming prompt
   - `params_fingerprint` from the `GenerateRequest` body
3. Look up an existing `CompletionCache` row.
4. If found:
   - Return the cached `output`
   - Mark the request as `cached=True` for logging/metrics
   - Compute token counts from the cached output
   - Log the request in `InferenceLog`
5. If not found:
   - Call the underlying model’s `generate(...)`
   - Compute token counts
   - Insert a new `CompletionCache` row
   - Insert an `InferenceLog` row
   - Return `cached=False`

This gives transparent deduplication: clients don’t need to know about caching, they
just see faster responses for repeated prompts.

#### 1.2 Batch requests: `/v1/generate/batch`

For `POST /v1/generate/batch`:

- The endpoint accepts a list of prompts plus shared generation parameters.
- It reuses the **same cache key logic**, but:
  - `params_fingerprint` is derived from the batch body excluding `prompts` and `model`.
  - Each prompt in the batch gets its own:
    - `prompt_hash`
    - completion
    - `CompletionCache` entry
    - `InferenceLog` entry

Behavior:

1. Compute a shared `params_fingerprint` for the batch.
2. For each prompt:
   - Check `CompletionCache` for `(model_id, prompt_hash, params_fingerprint)`.
   - If cached, return that `output` with `cached=True`.
   - If not cached:
     - Call `model.generate(...)`
     - Optionally insert a new `CompletionCache` row.
     - Log the item in `InferenceLog`.
3. Aggregate results into `BatchGenerateResponse`, with per-item fields:
   - `output`
   - `cached`
   - `prompt_tokens`
   - `completion_tokens`

The request-level `cached` flag (used in logging/metrics) is considered `True` only if
**all** prompts in the batch were cache hits.

### 2. Redis (Optional / Future Expansion)

Redis is wired into the application lifecycle but is not yet the primary cache for
completions. It currently behaves as:

- An optional dependency, controlled by configuration.
- A health signal for `/readyz` when `redis_enabled=true`.
- A future foundation for:
  - Hot-path, in-memory completion caching
  - Shared cache across multiple API replicas
  - Coordinating request coalescing or batching

Configuration knobs:

- `REDIS_ENABLED` / `redis_enabled` (boolean)
- `REDIS_URL` / `redis_url` (e.g. `redis://llm_redis:6379/0`)

Initialization model:

- On startup, the FastAPI app calls `init_redis()` and stores the client on
  `app.state.redis`.
- `/readyz` checks Redis only if `redis_enabled` is true.
- If Redis is disabled or unavailable (and `redis_enabled=false`), the app can still
  serve traffic with the database-backed cache.

This makes it possible to run:

- A minimal stack (no Redis, just Postgres cache), or
- A richer deployment with Redis as a high-speed layer on top of the DB.

### 3. Cache Invalidation and Model Changes

Cache entries are **tied to the model id and parameters**, not to model versioning or
weights beyond the `model_id` string.

Implications:

- If you fine-tune or swap weights but reuse the same `model_id`, old cache entries
  may no longer reflect the new model behavior.
- For production deployments, a typical pattern is:
  - Use versioned model IDs, e.g. `my-llm:v1`, `my-llm:v2`.
  - When you upgrade, change `MODEL_ID` (or `models.yaml` default) to the new id.
  - The cache is then naturally partitioned by `model_id`.

Manual invalidation strategies:

- Truncate the `CompletionCache` table:
  - when doing major model changes,
  - after large-scale configuration shifts,
  - or for cold-start performance testing.
- Optionally add management commands or admin endpoints to:
  - delete cache entries by model_id,
  - clear entries older than N days,
  - or export stats on cache hit rates.

### 4. Metrics and Visibility

Caching behavior is observable through Prometheus metrics, including:

- `llm_requests_total{route, model_id, cached=...}`
  - Allows you to track how many requests were served from cache vs live model.

- `llm_tokens_total{direction="prompt"|"completion", model_id=...}`
  - Token volume can be correlated with cache hits to see how much compute you’re saving.

You can inspect:

- cache effectiveness over time,
- per-model hit rates,
- and the impact of prompting patterns (e.g., repeated prompts in evals).

### 5. Configuration Summary

Key environment variables:

- `MODEL_ID`
  - Controls the default model for single-model mode.
  - Also affects cache isolation, since `model_id` is part of the key.

- `REDIS_ENABLED` / `redis_enabled`
  - Whether Redis is required in readiness checks.
  - When `false`, the system runs fully on Postgres-backed cache.

- `REDIS_URL` / `redis_url`
  - Connection URL for Redis if enabled.

The default, out-of-the-box behavior is:

- A robust, Postgres-backed completion cache.
- Optional Redis integration that can be enabled when you’re ready to introduce
  a distributed, in-memory caching layer.