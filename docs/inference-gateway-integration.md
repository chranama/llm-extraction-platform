# Inference Gateway Integration

These workflows run `llm-extraction-platform` behind the companion
`inference-serving-gateway` repository. They show how the backend behaves when
it sits behind an inference edge service across deterministic, live-model,
edge-control, containerized, and Kubernetes-shaped targets.

The gateway repository keeps its isolated mock-upstream workflows. This document
is the canonical place for the combined LLMEP plus gateway workflow because the
backend owns the API contract, worker lifecycle, policy behavior, admin traces,
and generated proof artifacts.

## Workflow Targets

| Workflow | Command | Model Backend | Runtime Shape | Artifact Directory |
|---|---|---|---|---|
| Deterministic baseline | `verify` | Fake deterministic profile | Host API/worker, Compose infra, host gateway | `proof/artifacts/joint_gateway/latest/` |
| Observability surface | `verify-observability` | Fake deterministic profile | Host API/worker, Compose infra/obs/OTel, host gateway | `proof/artifacts/joint_gateway/observability_latest/` |
| Edge controls | `verify-edge-controls` | Fake deterministic profile | Host API/worker, Compose infra, restarted host gateway variants | `proof/artifacts/joint_gateway/edge_controls_latest/` |
| Live llama.cpp extract | `verify-llama` | CPU-only containerized `llama.cpp` | LLMEP Compose extract stack, host gateway | `proof/artifacts/joint_gateway/llama_extract_latest/` |
| Containerized stack | `verify-containerized` | Fake deterministic profile | LLMEP API/worker containers, gateway container, Compose infra | `proof/artifacts/joint_gateway/containerized_latest/` |
| Kind smoke | `verify-kind` | Fake deterministic profile | Local kind cluster with LLMEP and gateway resources | `proof/artifacts/joint_gateway/kind_smoke_latest/` |

`verify` remains the compatibility alias for the original deterministic
baseline. Use the named commands when presenting a specific workflow.

## Host Runtime Shape

| Component | Runtime | Responsibility |
|---|---|---|
| LLMEP API server | Host process started with `uv` | Auth, schemas, sync extract, admin, traces, metrics |
| LLMEP worker | Host process started with `uv` | Async extract job execution |
| Postgres and Redis | LLMEP Docker Compose `infra-host` profile | Durable API keys, logs, traces, async job state |
| Prometheus and Grafana | Optional LLMEP Docker Compose `obs-host` profile | Local metrics scrape and dashboard inspection |
| OTel Collector and Jaeger | Optional LLMEP Docker Compose `otel-host` profile | Local trace export inspection |
| Inference gateway | Go binary built from sibling checkout | Edge routing, request identity, route controls, metrics, OTel propagation |
| Model backend | LLMEP deterministic fake model profile | Low-variance extract behavior for integration proof |

The deterministic host workflows use `proof/fixtures/models.gateway-proof.yaml`.
That keeps integration and edge-control proofs fast and low variance.

## Capability Set

The deterministic and observability targets verify:

- `POST /v1/extract` through the gateway.
- `POST /v1/extract/jobs` through the gateway.
- `GET /v1/extract/jobs/{job_id}` through the gateway.
- Gateway-to-backend request ID and trace ID propagation.
- LLMEP `EDGE_MODE=behind_gateway` behavior for trusted gateway headers.
- Backend metrics and gateway metrics from the same run.
- Admin trace inspection for sync and async extract requests.
- OTel export from gateway, backend, and worker into Jaeger when OTel is enabled.

The edge-control target verifies:

- Gateway route allow/deny behavior for sync and async extract routes.
- Gateway-owned unsupported route behavior for `/v1/generate`.
- Gateway-owned request-size rejection before the request reaches LLMEP.
- Backend auth failures are surfaced through the gateway without becoming
  gateway-owned errors.
- Gateway metrics are present for the edge-control cases.

The live llama target verifies the same sync/async extract path through the
gateway while LLMEP uses its CPU-only containerized `llama.cpp` Compose extract
runtime.

The containerized target verifies the same sync/async extract path while LLMEP
API, LLMEP worker, gateway, Postgres, and Redis run together on one local
Compose network.

The kind target verifies the Kubernetes-shaped local deployment with LLMEP API,
worker, gateway, Jaeger, and OTel resources in a local `kind` cluster.

It does not claim:

- Accelerated inference.
- Production throughput.
- `/v1/generate` support through the gateway.
- AWS, TLS, identity provider integration, or production ingress hardening.

## Run

From the LLMEP repository root:

```bash
tools/joint/inference_gateway_stack.sh preflight
tools/joint/inference_gateway_stack.sh up
tools/joint/inference_gateway_stack.sh status
tools/joint/inference_gateway_stack.sh proof
tools/joint/inference_gateway_stack.sh down
```

To start the deterministic stack, generate proof artifacts, and shut it down in
one command:

```bash
tools/joint/inference_gateway_stack.sh verify
```

Run named workflows:

```bash
tools/joint/inference_gateway_stack.sh verify-observability
tools/joint/inference_gateway_stack.sh verify-edge-controls
tools/joint/inference_gateway_stack.sh verify-llama
tools/joint/inference_gateway_stack.sh verify-containerized
tools/joint/inference_gateway_stack.sh verify-kind
```

The harness expects a sibling checkout by default:

```text
/path/to/career/
  llm-extraction-platform/
  inference-serving-gateway/
```

Set `ISG_REPO_ROOT` when the gateway checkout lives elsewhere:

```bash
ISG_REPO_ROOT=/path/to/inference-serving-gateway \
  tools/joint/inference_gateway_stack.sh up
```

## Useful Overrides

| Variable | Default | Use |
|---|---|---|
| `ISG_REPO_ROOT` | `../inference-serving-gateway` | Gateway checkout location |
| `JOINT_ARTIFACT_DIR` | `proof/artifacts/joint_gateway/latest` | Proof output location |
| `JOINT_OBSERVABILITY_ARTIFACT_DIR` | `proof/artifacts/joint_gateway/observability_latest` | Observability proof output location |
| `JOINT_EDGE_ARTIFACT_DIR` | `proof/artifacts/joint_gateway/edge_controls_latest` | Edge-control proof output location |
| `JOINT_LLAMA_ARTIFACT_DIR` | `proof/artifacts/joint_gateway/llama_extract_latest` | Live llama proof output location |
| `JOINT_CONTAINER_ARTIFACT_DIR` | `proof/artifacts/joint_gateway/containerized_latest` | Containerized proof output location |
| `JOINT_KIND_ARTIFACT_DIR` | `proof/artifacts/joint_gateway/kind_smoke_latest` | Kind proof output location |
| `JOINT_WITH_OBS` | `1` | Start Prometheus and Grafana |
| `JOINT_WITH_OTEL` | `1` | Start OTel Collector and Jaeger |
| `JOINT_RUN_TESTS` | `0` | Run all gateway tests during preflight |
| `JOINT_COMPOSE_PROJECT_NAME` | `llmep_joint_gateway` | Compose project name for isolated containers and volumes |
| `JOINT_BACKEND_PORT` | `18090` | Host LLMEP API port |
| `JOINT_GATEWAY_PORT` | `18091` | Host gateway port |
| `JOINT_PG_HOST_PORT` | `15433` | Host Postgres port |
| `JOINT_REDIS_HOST_PORT` | `16379` | Host Redis port |
| `JOINT_PROM_HOST_PORT` | `19091` | Host Prometheus port |
| `JOINT_GRAFANA_PORT` | `13000` | Host Grafana port |
| `JOINT_JAEGER_PORT` | `26688` | Host Jaeger port |
| `JOINT_LLAMA_ENV_FILE` | `.env.docker` | Env file for live llama Compose extract |
| `JOINT_LLAMA_PROJECT_NAME` | `llmep_joint_llama` | Compose project name for live llama workflow |
| `JOINT_CONTAINER_PROJECT_NAME` | `llmep_joint_containerized` | Compose project name for containerized workflow |
| `JOINT_KIND_CLUSTER` | `llm` | Local kind cluster used by the kind workflow |

The high default ports are intentional. They reduce collisions with LLMEP and
gateway isolated workflows.

## Inspect

After `up`, use:

```bash
tools/joint/inference_gateway_stack.sh status
```

Primary endpoints:

- Gateway: `http://127.0.0.1:18091`
- Backend: `http://127.0.0.1:18090`
- Prometheus: `http://127.0.0.1:19091`
- Grafana: `http://127.0.0.1:13000`
- Jaeger: `http://127.0.0.1:26688`

Runtime logs are under `.tmp/joint_gateway/logs/`.

## Workflow Notes

### Observability Surface

`verify-observability` runs the deterministic backend with OTel enabled and
writes gateway metrics, backend metrics, admin traces, execution logs, and
Jaeger trace exports. It is the best local workflow for demonstrating how a
reviewer can inspect a request across the edge service, backend service, and
async worker.

### Edge Controls

`verify-edge-controls` restarts the gateway under several configurations while
keeping the backend deterministic. It records allowed extract behavior,
unsupported `/v1/generate`, disabled extract routes, disabled async submit, an
oversized request, invalid backend auth, and gateway metrics.

### Live llama.cpp Extract

`verify-llama` starts LLMEP's promoted Compose extract stack with CPU-only
containerized `llama.cpp`, then places the host gateway in front of that API.
It requires the same prerequisites as `compose-extract`, including
`LLAMA_MODELS_DIR`, `LLAMA_MODEL_FILE`, and a local GGUF model file.

The model runtime is containerized, but it is CPU-only. This path demonstrates
real model-backed extraction correctness through the gateway, not accelerated
inference or production throughput.

### Containerized Stack

`verify-containerized` uses
`deploy/compose/docker-compose.gateway.yml` with the base Compose file. LLMEP
API, LLMEP worker, Postgres, Redis, and the gateway all run as containers on
one Compose network. This path uses the fake deterministic model profile first
so it can prove service networking and containerized orchestration without
model latency.

### Kind Smoke

`verify-kind` wraps the gateway repository's Kubernetes-shaped local stack and
copies the generated proof bundle back into LLMEP. The kind cluster is left
intact, but the workflow deletes the applied LLMEP/gateway resources after the
proof run.

## Proof Artifacts

`tools/joint/inference_gateway_stack.sh proof` writes to:

```text
proof/artifacts/joint_gateway/latest/
```

Representative files include:

- `extract.body.json`
- `extract_jobs.body.json`
- `job_status.body.json`
- `gateway.metrics.txt`
- `backend.metrics.txt`
- `sync_trace_detail.json`
- `async_trace_detail.json`
- `sync_logs.json`
- `async_logs.json`
- `sync_otel_trace.json`
- `async_otel_trace.json`
- `backend.log`
- `worker.log`
- `gateway.log`
- `manifest.json`
- `summary.md`

These artifacts are generated through the gateway repository's observability
pack helper while the runtime lifecycle remains owned by LLMEP.

The named workflows write equivalent manifests under their dedicated artifact
directories. `edge_controls_latest/` uses its own manifest because it records
gateway-owned rejection behavior rather than sync/async execution timelines.

## Failure Checks

If preflight fails on ports, either stop the colliding local service or override
the corresponding `JOINT_*_PORT` value.

If `/readyz` fails on the gateway, check the backend health endpoint first:

```bash
curl -fsS http://127.0.0.1:18090/healthz
curl -fsS http://127.0.0.1:18091/readyz
```

If trace identity is missing, verify that the backend environment file contains
`EDGE_MODE=behind_gateway`:

```bash
cat .tmp/joint_gateway/joint-gateway.env
```

If OTel artifacts are absent, verify that `JOINT_WITH_OTEL=1` and that Jaeger is
reachable at `http://127.0.0.1:26688`.

Always shut down the stack when finished:

```bash
tools/joint/inference_gateway_stack.sh down
```
