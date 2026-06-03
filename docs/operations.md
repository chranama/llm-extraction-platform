# Operations

This repository supports local host, compose, and Kubernetes-oriented workflows.
Some paths are lightweight validation paths; others require local infrastructure.

For ordered start, verification, log inspection, and shutdown commands, use
[`runbook.md`](runbook.md).

## Requirements

Common requirements:

- Python 3.12
- `uv`
- Docker for container and compose workflows
- Redis and Postgres for async extraction and integration workflows
- `kind` and Kubernetes tooling for local cluster checks

Model-backed runtime paths also require a configured local model profile.

## Health And Readiness

- `/healthz` is a liveness probe and should stay fast.
- `/readyz` checks DB, Redis, model readiness, policy state, and deployment metadata.
- `/modelz` exposes model readiness details.
- `/v1/models/status` provides a compact non-admin model status surface.

Relevant code:

- `server/src/llm_server/api/health.py`
- `server/src/llm_server/runtime/readiness.py`
- `server/src/llm_server/core/health_checks.py`

## Local Runtime

The root CLI exposes repo-level workflows through `llmctl`.

Example compose inspection command:

```bash
uv run llmctl --project-name llmep compose --env-override-file .env.docker ps
```

Deployment assets live in:

- `deploy/compose/`
- `deploy/docker/`
- `deploy/k8s/`
- `deploy/observability/`
- `deploy/proxy/`

## Async Extraction

Async extraction uses a separate worker process and durable job state.

Relevant code:

- `server/src/llm_server/application/submit_extract_job.py`
- `server/src/llm_server/application/process_extract_job.py`
- `server/src/llm_server/application/poll_extract_job.py`
- `server/src/llm_server/worker/extract_jobs.py`

Generated async evidence lives in `proof/artifacts/phase6_extract_async/`.

## Diagnostics

Useful surfaces:

- health/readiness endpoints for runtime state
- admin logs and trace detail endpoints
- Prometheus metrics from the server
- generated trace artifacts under `proof/artifacts/phase7_trace_inspection/`
- CI failure bundles uploaded by workflow jobs

Common failure areas:

- model profile not configured or model load mode disabled
- Redis unavailable for cache or queue behavior
- Postgres unavailable for durable job state
- policy artifact not present or not reloaded
- Kubernetes local cluster not running for `kind` checks
