# Architecture Deep Dive

This document explains the current architecture after the backend application/runtime refactor work.

The short version:

- `llm-extraction-platform` is still one backend service at deployment time
- internally, that backend is now more explicitly split into:
  - transport
  - application orchestration
  - domain/run state
  - runtime planning
  - observability/export seams

For the backend-internal view, read [Application Runtime Architecture](application-runtime-architecture.md).

## Repository-Level System Components

- `server/`: online runtime API for generate, extract, admin, and health/readiness surfaces
- `policy/`: policy decision engine and onboarding logic
- `eval/`: offline evaluation jobs and reporting pipeline
- `schemas/`: JSON schema specs for structured outputs
- `contracts/`: shared validation/types
- `integrations/`: repo-level end-to-end suites
- `ui/`: frontend client

## Backend Center Of Gravity

Inside `server/src/llm_server`, the architecture now reads more clearly as:

- `api/`
  - thin transport layer
  - request parsing, dependency injection, response shaping
- `application/`
  - request/use-case orchestration
  - sync extract, async submit, async poll
- `domain/`
  - durable concepts like `ExtractionRun`, `RunIdentity`, `RunOutcome`, and async job lifecycle
- `runtime/`
  - routing, prompt construction, generate-cap policy application, token counting, off-loop generation
- `observability/`
  - replay/export packaging from traces and execution logs
- `services/`
  - runtime adapters, enforcement helpers, queue/job services, and compatibility layers
- `telemetry/`
  - persisted trace and execution-log query surfaces

The main architectural change is that runtime behavior is no longer hidden behind a generic dependency layer.

## Runtime Request Paths

### Sync Extract

1. Client calls `POST /v1/extract`
2. `api/extract.py` acts as transport only
3. `application/run_extract.py` builds an `ExtractionRun`
4. `runtime/` logic decides:
   - model routing
   - prompt construction
   - generate-cap policy application
   - token accounting strategy
5. validation/repair and execution logging occur
6. trace + execution signals are persisted

### Async Extract

1. Client calls `POST /v1/extract/jobs`
2. `application/submit_extract_job.py` validates and persists the job
3. worker process claims and executes the job
4. `GET /v1/extract/jobs/{job_id}` polls current lifecycle state
5. submit, worker, and poll all correlate through:
   - `request_id`
   - `trace_id`
   - `job_id`

### Generate

`/v1/generate` still uses the older route shape, but its core runtime-planning pieces now come from `runtime/` instead of route-adjacent helper modules.

That means the architectural center of gravity is clearer even before generate is moved onto a fuller application-layer use case.

## Control-Plane Artifact Paths

### Eval -> Policy -> Runtime

1. `eval/` produces run artifacts
2. `policy/` onboarding consumes those artifacts
3. policy/model artifacts are written
4. `server/` reads those artifacts through `MODELS_YAML` and policy snapshots
5. runtime behavior changes deterministically

### Runtime Telemetry -> Replay Export

1. request trace events and execution logs are persisted
2. admin surfaces expose:
   - trace detail
   - execution-log detail
3. `observability/replay_cases.py` packages those into replay-oriented cases
4. `observability/regression_manifests.py` wraps them in a minimal regression-manifest shape

This does **not** replace `eval/`.

It creates the first architectural bridge from runtime observability to future replay/regression evaluation work.

## Gateway Integration Shape

The backend still supports two deployment identities:

- standalone backend
- gateway-backed backend through `EDGE_MODE=behind_gateway`

That external service boundary is described in:

- [Service Boundary: `inference-serving-gateway` Integration](service-boundary.inference-serving-gateway.md)
- [12) Trace Identity Contract](12-trace-identity-contract.md)

The internal backend refactor does not change that boundary.

It makes the backend side easier to explain:

- transport in `api/`
- orchestration in `application/`
- runtime decisions in `runtime/`
- replay/export packaging in `observability/`

## Deployment Shapes

- host server + docker infra
- docker server + docker infra
- docker server + host llama
- docker server + in-compose llama
- integrated local stack with gateway + backend
- local `kind` deployment for the integrated stack

See:

- [03-deployment-modes.md](03-deployment-modes.md)
- [Local Environment Contract](local-environment-contract.md)
- [Kind Deployment Contract](kind-deployment-contract.md)

## Design Constraints

- extraction contracts and schema validation remain explicit
- offline artifacts still influence runtime behavior intentionally
- trace identity must remain stable across standalone and gateway-backed modes
- async worker execution remains backend-owned
- replay/export is allowed to be partial:
  - a case may be exportable before it is fully replay-ready

## Best Follow-On Docs

- [Application Runtime Architecture](application-runtime-architecture.md)
- [Replay Export Seam](replay-export-seam.md)
- [12) Trace Identity Contract](12-trace-identity-contract.md)
- [Service Boundary: `inference-serving-gateway` Integration](service-boundary.inference-serving-gateway.md)
- [06) Eval Methodology](06-eval-methodology.md)
