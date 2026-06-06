# Operations

This repository supports local host, Compose, and Kubernetes-oriented workflows.
The promoted local runtime targets are documented in the runbook. Some paths are
lightweight validation paths; the Compose extract target requires a local GGUF
model and runs CPU-only containerized `llama.cpp`.

For ordered start, verification, log inspection, and shutdown commands, use
[`runbook.md`](runbook.md).

Use `llmctl preflight <target>` before a promoted path when you want a
non-mutating local setup check. Preflight validates binaries, Docker reachability
when required, deterministic Compose rendering, required env values, policy and
schema files, model-file paths, Kubernetes overlay rendering, evidence artifact
presence, and local port availability. It is a pre-start guardrail; `doctor`
remains the post-start diagnostic for a running Compose stack.

## Requirements

Common requirements:

- Python 3.12
- `uv`
- Docker for container and compose workflows
- Redis and Postgres for async extraction and integration workflows
- `kind` and Kubernetes tooling for local cluster checks

Model-backed runtime paths also require a configured local model profile. The
supported Compose extract profile is `compose-extract`; it uses
`policy_out/local_extract_allow.json` as the promoted local allow-policy
fixture. Deny decisions remain useful for policy demonstrations, but they are
not the default happy path.

The external model runtime path assumes the model server is already running and
reachable from the API container. This repo verifies the server boundary; it
does not own lifecycle or acceleration for that external runtime. Treat it as an
operational target rather than a canonical proof-backed target unless a separate
external runtime proof is generated.

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

The root CLI exposes curated repo-level workflows through `llmctl`.

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
- UI and local proxy surfaces for manual inspection
- generated trace artifacts under `proof/artifacts/phase7_trace_inspection/`
- policy/eval linkage artifacts under `proof/artifacts/phase9_policy_eval_linkage/`
- ops-surface artifacts under `proof/artifacts/phase10_ops_surface/`
  including Prometheus scrape state and Grafana dashboard population
- CI failure bundles uploaded by workflow jobs

Common failure areas:

- model profile not configured or model load mode disabled
- Redis unavailable for cache or queue behavior
- Postgres unavailable for durable job state
- policy artifact not present or not reloaded
- Kubernetes local cluster not running for `kind` checks
