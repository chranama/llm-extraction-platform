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

## Reviewer Smoke

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

## Kubernetes Smoke

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

Run:

```bash
uv run llmctl admin-trace
```

This executes `proof/generate_trace_inspection_proof.py`. The script starts a
proof server and worker, runs sync and async extract requests, fetches admin
trace detail for both flows, and writes ordered trace artifacts under
`proof/artifacts/phase7_trace_inspection/`.

## UI/Observability/Proxy

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
