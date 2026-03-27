# Application Runtime Architecture

This document explains the backend-internal architecture of `llm-extraction-platform` after the application/runtime refactor work.

Its purpose is simple:

- make it easy to answer “where does runtime behavior live?”
- make the backend legible as an AI application runtime rather than only a route collection

## Why This Layering Exists

The backend still deploys as one service, but the code now distinguishes between:

- HTTP transport concerns
- use-case orchestration
- run/domain state
- runtime planning
- observability/export seams

That separation matters because the backend now needs to support:

- standalone operation
- gateway-backed operation
- sync and async extract flows
- richer observability and replay export

## Current Internal Layers

### `api/`

Role:

- FastAPI transport layer
- request parsing
- dependency injection
- response shaping

Examples:

- [api/extract.py](../server/src/llm_server/api/extract.py)
- [api/generate.py](../server/src/llm_server/api/generate.py)
- [api/admin.py](../server/src/llm_server/api/admin.py)

Expected rule:

- route files should not own the full business/runtime story

### `application/`

Role:

- use-case orchestration
- sync extract orchestration
- async submit/poll orchestration
- request-scoped session handling where appropriate

Examples:

- [application/run_extract.py](../server/src/llm_server/application/run_extract.py)
- [application/submit_extract_job.py](../server/src/llm_server/application/submit_extract_job.py)
- [application/poll_extract_job.py](../server/src/llm_server/application/poll_extract_job.py)

Expected rule:

- application modules coordinate the flow
- they should call into runtime/services rather than re-implement low-level helpers

### `domain/`

Role:

- stable runtime concepts
- run identity
- run outcome
- async job lifecycle

Examples:

- [domain/runs.py](../server/src/llm_server/domain/runs.py)
- [domain/outcomes.py](../server/src/llm_server/domain/outcomes.py)
- [domain/jobs.py](../server/src/llm_server/domain/jobs.py)

Key concept:

- `ExtractionRun` is now the central backend unit for extract behavior

That makes the system read more like:

- “a request creates or updates a run”

instead of:

- “a route plus helper modules happen to emit logs”

### `runtime/`

Role:

- model routing
- readiness/routing helpers
- prompt construction
- generate-cap policy application
- off-loop generation execution
- token counting strategy

Examples:

- [runtime/routing.py](../server/src/llm_server/runtime/routing.py)
- [runtime/prompts.py](../server/src/llm_server/runtime/prompts.py)
- [runtime/generation.py](../server/src/llm_server/runtime/generation.py)

This is the clearest answer to:

- “where is runtime planning decided?”

### `observability/`

Role:

- package traces and execution logs into replay-oriented cases
- create minimal regression-manifest outputs

Examples:

- [observability/replay_cases.py](../server/src/llm_server/observability/replay_cases.py)
- [observability/regression_manifests.py](../server/src/llm_server/observability/regression_manifests.py)

This is the first explicit bridge from:

- runtime observability

to:

- future replay/regression evaluation

### `services/`

Role:

- lower-level adapters and operations
- queue and worker logic
- enforcement helpers
- compatibility layers retained during the refactor

Examples:

- [services/extract_execution.py](../server/src/llm_server/services/extract_execution.py)
- [services/extract_jobs.py](../server/src/llm_server/services/extract_jobs.py)
- [services/llm_runtime](../server/src/llm_server/services/llm_runtime)

Important nuance:

- route-specific dependency wiring now lives under `api/dependencies/`
- non-route code should not depend on route-dependency packages
- `services/` is now intended as execution substrate, not as a second application layer

## Directory Map

The current `server/src/llm_server` layout is:

```text
llm_server/
  api/
    dependencies/
  application/
  core/
  db/
  domain/
  io/
  observability/
  reports/
  runtime/
  services/
    backends/
    extract_support/
    limits/
    llm_runtime/
  telemetry/
  worker/
```

How to read that structure:

- `api/`
  - transport surfaces only
  - route files and route-only dependency wiring
- `application/`
  - use-case orchestration
  - sync extract and async submit/poll entrypoints
- `domain/`
  - stable runtime concepts like runs, outcomes, and job lifecycle
- `runtime/`
  - request-time planning and control-plane decisions
  - routing, readiness interpretation, prompt construction, runtime policy application
- `observability/`
  - derived artifacts and exports built from runtime signals
  - replay cases and regression manifests
- `core/`
  - shared primitives and cross-cutting helpers
  - config, errors, metrics, request settings, health checks, schema loading
- `db/`
  - persistence models and session wiring
- `io/`
  - artifact/file boundaries such as policy snapshots and runtime SLO artifact I/O
- `telemetry/`
  - persisted raw runtime signals and query helpers
  - traces, execution-log queries, telemetry types
- `services/`
  - execution substrate and backend machinery
  - this is not the main application layer
- `worker/`
  - separate background-process entrypoints

Important subpackages:

- `api/dependencies/`
  - FastAPI dependency surfaces for route modules only
- `services/backends/`
  - concrete backend adapters (`transformers`, `llama.cpp`, fake, remote/OpenAI-compatible)
- `services/extract_support/`
  - low-level extract execution helpers such as JSON parsing, truncation detection, and error staging
- `services/limits/`
  - request limiting, early reject, and generate-gating substrate
- `services/llm_runtime/`
  - model runtime machinery: registry build/load, model state, inference cache/log plumbing

The intended architectural reading is:

- `api + application + domain + runtime + observability`
  - the AI application runtime and control-plane layer
- `core + db + io + telemetry + services`
  - the supporting substrate

## Extract Flow After The Refactor

### Sync extract

1. transport enters through [api/extract.py](../server/src/llm_server/api/extract.py)
2. orchestration enters [application/run_extract.py](../server/src/llm_server/application/run_extract.py)
3. a new `ExtractionRun` is built
4. runtime decisions come from:
   - [runtime/routing.py](../server/src/llm_server/runtime/routing.py)
   - [runtime/prompts.py](../server/src/llm_server/runtime/prompts.py)
   - [runtime/generation.py](../server/src/llm_server/runtime/generation.py)
5. execution and validation complete in the service layer
6. telemetry and execution logs become queryable/admin-visible

### Async extract

1. submit enters [application/submit_extract_job.py](../server/src/llm_server/application/submit_extract_job.py)
2. the worker executes through [services/extract_jobs.py](../server/src/llm_server/services/extract_jobs.py)
3. poll enters [application/poll_extract_job.py](../server/src/llm_server/application/poll_extract_job.py)
4. all three stages share the same correlation model:
   - `request_id`
   - `trace_id`
   - `job_id`

## Relationship To Gateway Integration

This internal refactor does **not** move extraction semantics into the gateway.

The gateway boundary stays:

- gateway owns edge/runtime concerns
- backend owns application semantics, async execution, trace inspection, and replay/export packaging

See:

- [service-boundary.inference-serving-gateway.md](service-boundary.inference-serving-gateway.md)

## Relationship To Future Work

This architecture is meant to support:

- OpenTelemetry rollout on top of a cleaner identity/runtime model
- AWS deployment work on top of a clearer operational surface
- later eval modernization work that can consume replay-oriented exports

## Reviewer Shortcut

If you only inspect a few files, use these:

- [api/extract.py](../server/src/llm_server/api/extract.py)
- [application/run_extract.py](../server/src/llm_server/application/run_extract.py)
- [domain/runs.py](../server/src/llm_server/domain/runs.py)
- [runtime/routing.py](../server/src/llm_server/runtime/routing.py)
- [runtime/generation.py](../server/src/llm_server/runtime/generation.py)
- [observability/replay_cases.py](../server/src/llm_server/observability/replay_cases.py)
