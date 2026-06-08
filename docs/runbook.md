# Runbook

This runbook promotes a small set of supported runtime targets. The root
`llmctl` commands start and verify the happy paths; normal tools such as
`docker compose`, `kubectl`, `curl`, and the proof scripts remain the preferred
inspection tools.

Use [`operations.md`](operations.md) for runtime concepts and failure modes.

## Supported Runtime Targets

| Target | Runtime Shape | Model Backend | What It Proves | What It Does Not Prove |
|---|---|---|---|---|
| Reviewer Smoke | API server plus local Compose infra | Fake deterministic backend | API startup, health/readiness, schemas, generate, sync extract, and basic auth/gating with low runtime variability | Real model quality or model-serving performance |
| Compose Extract | API server, Postgres, Redis, worker, and llama-server in Docker Compose | CPU-only containerized `llama.cpp` serving SmolLM2 | Real model-backed generate/extract, sync extract, async extract, durable job state, policy/capability surfaces, and traceable runtime behavior | Accelerated inference, production throughput, GPU scheduling, or high availability |
| External Model Runtime | Containerized API server calls a model runtime outside the server container | Host `llama.cpp` or Docker-managed model runtime | Production-relevant model-serving boundary and external backend integration | Fully containerized accelerated inference on Docker for Mac |
| Joint Inference Gateway | Host API server and worker, Compose infra, and sibling Go gateway | Fake deterministic backend | Backend behavior behind an inference edge service, sync and async extract forwarding, request identity, traces, and metrics across both services | Real model quality, `/v1/generate` through the gateway, AWS deployment, TLS, or production ingress |
| Joint Live Llama Extract | LLMEP Compose extract stack with sibling Go gateway | CPU-only containerized `llama.cpp` | Real model-backed sync and async extraction through the gateway | Accelerated inference, production throughput, or cloud model serving |
| Joint Edge Controls | Host API server and worker with restarted gateway variants | Fake deterministic backend | Gateway-owned route policy, unsupported route, body-size rejection, backend auth pass-through, and metrics | Real model behavior or load testing |
| Joint Containerized Stack | LLMEP API/worker containers, gateway container, and Compose infra | Fake deterministic backend | Same local Compose network for backend, worker, gateway, Postgres, and Redis | Real model quality or production orchestration |
| Joint Containerized Live Llama | LLMEP API/worker containers, gateway container, llama.cpp container, and Compose infra | CPU-only containerized `llama.cpp` | Real model-backed sync and async extraction through a fully containerized local joint stack | Accelerated inference, production throughput, cloud ingress, or high availability |
| Joint Resilience | LLMEP API/worker containers, gateway container, Postgres, and Redis with controlled interruptions | Fake deterministic backend | Bounded local failure behavior, observable degradation, and recovery after component restart | High availability, autoscaling, zero downtime, or cloud failover |
| Joint Kind Live Llama | Local kind deployment using LLMEP, gateway, worker, OTel/Jaeger, and llama-server resources | CPU-only containerized `llama.cpp` with a host-mounted GGUF model | Kubernetes-shaped joint deployment with real model-backed sync/async extraction, gateway forwarding, traces, logs, and model-runtime artifacts | Accelerated inference, production throughput, cloud ingress, AWS, TLS, or production HA |
| Kubernetes Smoke | Local `kind` deployment | Fake generate-only backend | Kubernetes deployability, readiness, services, and extract-disabled capability gating | Full extraction workflow or real model serving |
| Policy/Eval Linkage | Host proof server with Postgres/Redis and generated eval fixtures | Fake deterministic backend | Eval artifact to policy decision flow, admin policy reload, and runtime extract allow/deny behavior | Model quality or full evaluation dataset coverage |
| Admin/Trace | Host proof server with Postgres/Redis and async worker | Fake deterministic backend | Admin trace inspection for sync and async extract, including worker lineage | Distributed tracing export or external telemetry compliance |
| UI/Observability/Proxy | Compose API, UI, Prometheus, Grafana, and nginx proxy | Fake deterministic backend | Direct and proxied access to API, UI, and observability surfaces | Production authentication, TLS, or cloud ingress behavior |
| Evidence Validation | Saved proof artifacts and validation scripts | Depends on artifact group | Repo claims are backed by inspectable runtime evidence | Live production behavior unless regenerated against a live target |

## Prerequisites

Common prerequisites:

```bash
uv sync --extra dev
docker info
```

For `compose-extract`, configure a local env override file. The default local
file is `.env.docker`; it must include:

```bash
API_KEY=<local-api-key>
LLAMA_MODELS_DIR=/path/to/gguf-model-directory
LLAMA_MODEL_FILE=/models/path/inside/mounted-directory.gguf
```

The model path is mounted into the `llama_server` container at `/models`.
`LLAMA_N_GPU_LAYERS=0` is the supported default. This path demonstrates
CPU-only, real-model extraction correctness, not accelerated inference.

For the joint live kind workflow, the same `.env.docker` model settings are
used by the gateway repository's kind harness. `LLAMA_MODELS_DIR` is mounted
into the kind control-plane node at `/models` when the cluster is created. If
the `llm` kind cluster already exists without that mount, delete and recreate
the cluster before running `verify-kind`.

## Preflight

Run preflight before starting a target when you want to catch local setup
problems without starting containers, applying migrations, seeding API keys, or
generating proof artifacts.

```bash
uv run llmctl --project-name llmep --env-override-file .env.docker preflight smoke
uv run llmctl --project-name llmep --env-override-file .env.docker preflight compose-extract
uv run llmctl --project-name llmep --env-override-file .env.docker preflight kind-live
uv run llmctl preflight evidence --json
```

Supported preflight targets are `smoke`, `compose-extract`, `external-model`,
`kind-smoke`, `kind-live`, `policy-eval`, `admin-trace`, `ops-surface`,
`evidence`, and `all`.

## Reviewer Smoke

Preflight:

```bash
uv run llmctl --project-name llmep --env-override-file .env.docker preflight smoke
```

Run:

```bash
uv run llmctl --project-name llmep --env-override-file .env.docker smoke
```

This starts the API server with the fake `test` model profile, applies database
migrations, seeds local API keys, and verifies:

- `/healthz`
- `/readyz`
- `/v1/schemas`
- `/v1/schemas/sroie_receipt_v1`
- `/v1/generate`
- `/v1/extract`

Inspect:

```bash
docker compose -p llmep ps
docker compose -p llmep logs --tail=200 server
```

Stop:

```bash
uv run llmctl --project-name llmep --env-override-file .env.docker stop
```

## Compose Extract

Preflight:

```bash
uv run llmctl --project-name llmep --env-override-file .env.docker preflight compose-extract
```

Run:

```bash
uv run llmctl --project-name llmep --env-override-file .env.docker compose-extract
```

This starts:

- `postgres`
- `redis`
- `llama_server`
- `server_llama`
- `worker_llama`

The path uses `config/models.yaml` profile `compose-extract`. The server talks
to the `llama_server` service over Compose DNS at `http://llama_server:8080`.
It also uses `policy_out/local_extract_allow.json` so the real-model extract
path is intentionally open for this workflow. Keep `policy_out/latest.json`
available for policy-deny/failure demonstrations; it is not the promoted
Compose extract default.

The command verifies:

- `/healthz`
- `/readyz`
- `/v1/models/status`
- `/v1/generate`
- `/v1/extract`
- `/v1/extract/jobs` with worker-backed completion

Inspect:

```bash
docker compose -p llmep ps
docker compose -p llmep logs --tail=200 server_llama
docker compose -p llmep logs --tail=200 llama_server
docker compose -p llmep logs --tail=200 worker_llama
```

Manual sync extract probe:

```bash
export API_BASE="http://localhost:8000"
export API_KEY="<local-api-key>"

curl -fsS -X POST "${API_BASE}/v1/extract" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ${API_KEY}" \
  --data '{
    "schema_id": "sroie_receipt_v1",
    "text": "ACME STORE\n123 MAIN ST\nDATE: 2024-03-10\nTOTAL: $42.18",
    "max_new_tokens": 512,
    "temperature": 0.0,
    "cache": false,
    "repair": true
  }' | python -m json.tool
```

Stop:

```bash
uv run llmctl --project-name llmep --env-override-file .env.docker stop
```

## External Model Runtime

Use this path when the model runtime is running outside the server container,
for example host `llama.cpp` or Docker Model Runner. This repo assumes that
external runtime is already started and reachable at `LLAMA_SERVER_URL`; setup,
acceleration, and lifecycle for that model server are intentionally external to
this target.

Preflight:

```bash
uv run llmctl --project-name llmep --env-override-file .env.docker preflight external-model
```

Run:

```bash
uv run llmctl --project-name llmep --env-override-file .env.docker external-model
```

This starts the containerized API server and routes model traffic through
`LLAMA_SERVER_URL`, defaulting to `http://host.docker.internal:8080` for the
host-routed llama path.

This target proves an external model-serving boundary. It does not claim that
the model runtime is containerized or accelerated inside the Compose stack. It
is an operational target, not part of the canonical proof bundle, because the
model runtime lifecycle is external to this repository.

## Joint Inference Gateway

Use this path to run LLMEP behind the companion `inference-serving-gateway`
checkout. This target is owned by LLMEP because the backend owns API contracts,
worker state, policy behavior, admin traces, and the integrated proof artifacts.

Run:

```bash
tools/joint/inference_gateway_stack.sh preflight
tools/joint/inference_gateway_stack.sh up
tools/joint/inference_gateway_stack.sh status
tools/joint/inference_gateway_stack.sh proof
tools/joint/inference_gateway_stack.sh down
```

Use `tools/joint/inference_gateway_stack.sh verify` to run startup, proof
generation, and shutdown as one sequence.

Named joint workflows:

```bash
tools/joint/inference_gateway_stack.sh verify-observability
tools/joint/inference_gateway_stack.sh verify-edge-controls
tools/joint/inference_gateway_stack.sh verify-llama
tools/joint/inference_gateway_stack.sh verify-containerized
tools/joint/inference_gateway_stack.sh verify-containerized-llama
tools/joint/inference_gateway_stack.sh verify-resilience
tools/joint/inference_gateway_stack.sh verify-kind
```

The default workflow uses the deterministic `gateway-proof` model profile.
`verify-llama` uses the promoted CPU-only Compose extract path with
containerized `llama.cpp`. `verify-containerized` runs LLMEP and the gateway as
containers on one Compose network. `verify-containerized-llama` combines those
two dimensions by running LLMEP, the worker, the gateway, and `llama.cpp` as
containers on one Compose network. `verify-resilience` interrupts the
containerized fake-backend joint stack to capture degradation and recovery
behavior. `verify-kind` runs the promoted Kubernetes-shaped local path with a
live CPU-only `llama.cpp` model server in kind, then leaves the kind cluster
intact while deleting applied resources.

For the older deterministic Kubernetes smoke path, run:

```bash
JOINT_KIND_WORKFLOW=fake \
JOINT_KIND_ARTIFACT_DIR=proof/artifacts/joint_gateway/kind_smoke_latest \
  tools/joint/inference_gateway_stack.sh verify-kind
```

Set `JOINT_KIND_ENV_FILE=/path/to/env` when the live kind workflow should read
model settings from a file other than `.env.docker`.

Set `JOINT_KIND_CLUSTER=llm-live` to run the live kind workflow in a separate
cluster instead of recreating an existing `llm` cluster that lacks the `/models`
mount.

See [Inference Gateway Integration](inference-gateway-integration.md) for the
runtime shape, supported capability set, overrides, and artifact inventory.

## Kubernetes Smoke

Preflight:

```bash
uv run llmctl preflight kind-smoke
```

Run:

```bash
uv run llmctl kind-smoke
```

This executes `proof/generate_k8s_kind_proof.py`. The script creates or reuses
the `llm` kind cluster, builds and loads `llm-server:dev`, applies the
`local-generate-only` overlay, runs smoke checks, and writes artifacts under
`proof/artifacts/phase5_k8s_kind/`.

Inspect manually:

```bash
kubectl -n llm get all
kubectl -n llm logs deployment/api --tail=200
```

Tear down:

```bash
uv run llmctl k8s delete-local-generate-only
uv run llmctl k8s kind-down
```

## Policy/Eval Linkage

Preflight:

```bash
uv run llmctl preflight policy-eval
```

Run:

```bash
uv run llmctl policy-eval
```

This executes `proof/generate_policy_eval_linkage_proof.py`. The script writes
small deterministic passing and failing extract eval summaries, runs the real
policy CLI against both summaries, starts a proof server, and verifies:

- passing eval produces an allow policy
- failing eval produces a deny policy
- `/v1/admin/policy` exposes the active policy snapshot
- `/v1/admin/policy/reload` reloads the policy artifact from disk
- `/v1/extract` succeeds under the allow policy and is blocked under the deny policy

Artifacts are written under
`proof/artifacts/phase9_policy_eval_linkage/`.

This path proves eval-artifact-to-policy-to-runtime linkage. It does not claim a
full benchmark or dataset evaluation run.

## Admin/Trace

Preflight:

```bash
uv run llmctl preflight admin-trace
```

Run:

```bash
uv run llmctl admin-trace
```

This executes `proof/generate_trace_inspection_proof.py`. The script starts a
proof server and worker, runs sync and async extract requests, fetches admin
trace detail for both flows, and writes ordered trace artifacts under
`proof/artifacts/phase7_trace_inspection/`.

## UI/Observability/Proxy

Preflight:

```bash
uv run llmctl preflight ops-surface
```

Run:

```bash
uv run llmctl ops-surface
```

This executes `proof/generate_ops_surface_proof.py`. The script starts the
reviewer smoke API path, then starts the UI, Prometheus, Grafana, and local
nginx proxy profiles. It verifies direct and proxied access to:

- API health
- UI index
- Prometheus readiness
- Grafana health
- Prometheus API scrape target state
- Grafana datasource and dashboard population

Artifacts are written under `proof/artifacts/phase10_ops_surface/`.

## Evidence Validation

Preflight:

```bash
uv run llmctl preflight evidence
```

Validate saved artifacts:

```bash
uv run llmctl evidence
```

Regenerate the canonical bundle:

```bash
uv run llmctl evidence --regenerate
```

The canonical regeneration refreshes the local proof paths that do not require a
machine-specific GGUF model. It validates the saved phase 8 artifact paths, but
the phase 8 Compose llama proof is generated separately because it needs a local
model file.

Generate the Compose llama extract proof when the local GGUF model and Docker
runtime are available:

```bash
python proof/generate_compose_llama_extract_proof.py
```

The Compose llama proof writes artifacts under
`proof/artifacts/phase8_compose_llama_extract/`.

## Doctor

Inspect a running Compose stack:

```bash
uv run llmctl --project-name llmep --env-override-file .env.docker doctor
```

The doctor checks Compose rendering, port status, health/readiness endpoints,
model status, schema availability, and an extract probe when possible.

## Stop

Stop the Compose project while preserving named volumes:

```bash
uv run llmctl --project-name llmep --env-override-file .env.docker stop
```

Stop and remove named volumes:

```bash
uv run llmctl --project-name llmep --env-override-file .env.docker stop --volumes
```

## Troubleshooting

If `compose-extract` fails before startup, check `LLAMA_MODELS_DIR` and
`LLAMA_MODEL_FILE`. The CLI validates that the host GGUF file exists before it
starts Compose.

If readiness fails, inspect:

```bash
curl -fsS http://localhost:8000/readyz | python -m json.tool
docker compose -p llmep logs --tail=200 server_llama
docker compose -p llmep logs --tail=200 llama_server
```

If sync extract succeeds but async extract stalls, inspect the worker:

```bash
docker compose -p llmep logs --tail=200 worker_llama
curl -fsS http://localhost:8000/v1/models/status \
  -H "X-API-Key: ${API_KEY}" | python -m json.tool
```
