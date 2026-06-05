# Architecture

LLM Extraction Platform is organized around a runtime API service, policy and
evaluation tooling, shared contracts, deployment assets, and generated runtime
artifacts.

## Components

- `server/`: FastAPI service for generate, extract, async jobs, admin, health,
  readiness, model status, and trace inspection.
- `policy/`: policy decision engine that reads evaluation and SLO artifacts and
  writes decisions consumed by the runtime service.
- `eval/`: evaluation workflows and scoring code for model behavior and
  extraction quality.
- `contracts/`: shared Pydantic models for internal runtime and artifact
  contracts.
- `schemas/`: JSON schemas used by extraction workflows.
- `integrations/`: cross-service test lanes for live server, eval, policy, and
  end-to-end flows.
- `deploy/`: Docker, compose, Kubernetes, observability, and proxy assets.
- `proof/`: generated runtime artifacts and validation scripts.
- `ui/`: frontend surface for interacting with and inspecting the service.

## Runtime Flows

### Generate

`/v1/generate` resolves the target model, checks model capability, applies
runtime policy controls, optionally uses cache state, and records request
metadata for diagnostics.

Important code areas:

- `server/src/llm_server/api/generate.py`
- `server/src/llm_server/runtime/`
- `server/src/llm_server/services/`

### Sync Extract

`/v1/extract` loads the requested schema, checks extraction capability, runs the
model-backed extraction flow, validates the structured output, and returns a
typed response.

Important code areas:

- `server/src/llm_server/api/extract.py`
- `server/src/llm_server/application/run_extract.py`
- `server/src/llm_server/core/schema_registry.py`
- `schemas/`

### Async Extract

`/v1/extract/jobs` submits extraction work, stores durable job state, and returns
a polling path. A separate worker executes queued work and updates final job
state.

Important code areas:

- `server/src/llm_server/application/submit_extract_job.py`
- `server/src/llm_server/application/process_extract_job.py`
- `server/src/llm_server/application/poll_extract_job.py`
- `server/src/llm_server/worker/extract_jobs.py`

### Evaluation To Policy

Evaluation and SLO artifacts are produced outside the runtime request path. The
policy package reads those artifacts and writes runtime decisions that the server
can reload through the admin policy endpoints. The proof bundle includes an
eval-to-policy path that shows passing and failing eval summaries producing
different runtime extract behavior after admin reload.

Important code areas:

- `eval/src/llm_eval/`
- `policy/src/llm_policy/`
- `server/src/llm_server/io/policy_decisions.py`

### Local Ops Surface

The Compose ops path exposes the API, UI, Prometheus, and Grafana directly and
through a local nginx proxy. Its evidence captures API scrape state and Grafana
dashboard population so the observability surface is more than a reachability
check. This is a local inspection surface, not a production ingress hardening
claim.

Important code areas:

- `deploy/compose/docker-compose.yml`
- `deploy/proxy/nginx/nginx.compose.conf`
- `deploy/observability/`
- `ui/`

## Design Boundaries

- API handlers live under `server/src/llm_server/api/`.
- Application use cases live under `server/src/llm_server/application/`.
- Domain objects live under `server/src/llm_server/domain/`.
- Runtime model, capability, routing, and readiness logic lives under
  `server/src/llm_server/runtime/`.
- Persistence lives under `server/src/llm_server/db/`.
- Trace and replay inspection live under `server/src/llm_server/telemetry/` and
  `server/src/llm_server/observability/`.

The main architectural risk is scope size: this repository contains multiple
subsystems. The package boundaries are intended to make those responsibilities
inspectable without requiring one large application file.
