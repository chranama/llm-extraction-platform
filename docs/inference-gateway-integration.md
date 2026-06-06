# Inference Gateway Integration

This workflow runs `llm-extraction-platform` behind the companion
`inference-serving-gateway` repository. It is the first supported joint runtime
target for showing how the backend behaves when it sits behind an inference edge
service.

The gateway repository keeps its isolated mock-upstream workflows. This document
is the canonical place for the combined LLMEP plus gateway workflow because the
backend owns the API contract, worker lifecycle, policy behavior, admin traces,
and generated proof artifacts.

## Runtime Shape

| Component | Runtime | Responsibility |
|---|---|---|
| LLMEP API server | Host process started with `uv` | Auth, schemas, sync extract, admin, traces, metrics |
| LLMEP worker | Host process started with `uv` | Async extract job execution |
| Postgres and Redis | LLMEP Docker Compose `infra-host` profile | Durable API keys, logs, traces, async job state |
| Prometheus and Grafana | Optional LLMEP Docker Compose `obs-host` profile | Local metrics scrape and dashboard inspection |
| OTel Collector and Jaeger | Optional LLMEP Docker Compose `otel-host` profile | Local trace export inspection |
| Inference gateway | Go binary built from sibling checkout | Edge routing, request identity, route controls, metrics, OTel propagation |
| Model backend | LLMEP deterministic fake model profile | Low-variance extract behavior for integration proof |

The supported default uses `proof/fixtures/models.gateway-proof.yaml`. That
keeps the joint workflow deterministic and fast. Real-model extraction remains
covered by the Compose extract path in the main runbook, where the model backend
is owned by LLMEP rather than by the gateway.

## Capability Set

The joint target verifies:

- `POST /v1/extract` through the gateway.
- `POST /v1/extract/jobs` through the gateway.
- `GET /v1/extract/jobs/{job_id}` through the gateway.
- Gateway-to-backend request ID and trace ID propagation.
- LLMEP `EDGE_MODE=behind_gateway` behavior for trusted gateway headers.
- Backend metrics and gateway metrics from the same run.
- Admin trace inspection for sync and async extract requests.
- Optional OTel export from gateway, backend, and worker into Jaeger.

It does not claim:

- Real-model extraction quality.
- `llama.cpp` model serving behind the gateway.
- `/v1/generate` through the gateway.
- All-in-container orchestration.
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

To start the stack, generate proof artifacts, and shut it down in one command:

```bash
tools/joint/inference_gateway_stack.sh verify
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
