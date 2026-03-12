# Deployment Modes

This document describes the deployment shapes actively used in this repository and how they differ.

## Mode overview

### 1) Host server + host infra
- Server runs on host Python environment.
- Postgres/Redis run in compose via `infra-host` profile.
- Typical for local iteration and debugging.

### 2) Docker server + in-compose infra
- Server runs in container (`server` profile).
- Postgres/Redis run as compose services (`infra` profile).
- Typical for container parity and integration checks.

### 3) Docker server + llama backend variants
- `server-llama`: server container uses in-compose `llama_server`.
- `server-llama-host`: server container targets host llama-server (`host.docker.internal`).
- Used by demo and integration scenarios that validate external LLM runtime wiring.

### 4) Kubernetes kind generate-only proof
- Server runs in a local `kind` cluster.
- Uses `deploy/k8s/overlays/local-generate-only`.
- Canonical proof validates rollout, health, generate behavior, and extract disablement.
- `deploy/k8s/overlays/prod-gpu-full` is render-validated scaffold only.

### 5) Host async extraction proof
- Server runs on the host Python environment.
- Postgres and Redis run via compose `infra-host`.
- A separate worker process consumes extract jobs from Redis and writes durable state to Postgres.
- Canonical proof validates submission, status polling, worker execution, and final successful result.

## Configuration contract

Core environment controls:
- `APP_PROFILE`: selects app config profile from `config/server.yaml`
- `MODELS_YAML`: points to models config file
- `MODELS_PROFILE`: selects model profile inside models YAML
- `DATABASE_URL`, `REDIS_URL`
- `POLICY_DECISION_PATH` (optional runtime policy overlay)
- `SCHEMAS_DIR` (defaults to repo-level model output schema directory)

Defaults are composed from:
- `config/compose-defaults.yaml`
- `deploy/compose/docker-compose.yml`

## Capabilities across modes

Generate is expected across all supported modes.

Extract is conditional and depends on:
- selected model capability,
- profile wiring (`MODELS_YAML` / `MODELS_PROFILE`),
- policy/runtime switches.

The canonical proof path is the extract-gate demo: [Project Demos](02-project-demos.md).

Kubernetes adds a second canonical deployment proof:
- live `kind` runtime proof for the generate-only service
- render-only scaffold proof for `prod-gpu-full`

Async extraction adds a workflow proof:
- `POST /v1/extract/jobs` returns `202`
- `GET /v1/extract/jobs/{job_id}` exposes durable job state
- a separate worker process executes the job and writes the final result

## Canonical compose usage

Bring up docker infra + server + UI using `llmctl` defaults composition:

```bash
uv run llmctl --project-name llmep \
  compose --defaults-profile docker \
  --env-override-file .env.docker \
  up --profiles infra server ui -d --build --remove-orphans
```

For llama-backed docker server flows:

```bash
uv run llmctl --project-name llmep \
  compose --defaults-profile docker+llama+models-llama \
  --env-override-file .env.docker \
  up --profiles infra llama server-llama ui -d --build --remove-orphans
```

For host-infra-only setup (server on host):

```bash
uv run llmctl --project-name llmep \
  compose --defaults-profile host \
  --env-override-file .env.docker \
  up --profiles infra-host -d --remove-orphans
```

## Operational diagnostics

When behavior differs across modes, check in this order:
1. `/readyz`
2. `/modelz` and `/v1/models`
3. active `MODELS_YAML` and `MODELS_PROFILE`
4. policy overlay source (`POLICY_DECISION_PATH`)
5. compose service logs for server + dependencies

For Kubernetes proof runs, also check:
6. `kubectl -n llm get pods -o wide`
7. `kubectl -n llm get svc`
8. `proof/artifacts/phase5_k8s_kind/k8s_smoke.log`

For async extraction proof runs, also check:
9. `proof/artifacts/phase6_extract_async/async_submit_response.json`
10. `proof/artifacts/phase6_extract_async/async_job_final.json`
11. `proof/artifacts/phase6_extract_async/async_worker_log.txt`

## Demo-specific note

For offline extract-gate demo runs, clear policy overlay unless explicitly testing it:

```bash
export POLICY_DECISION_PATH=""
```

This isolates capability behavior to onboarding artifacts and model profile wiring.

## Related docs

- [Project Demos](02-project-demos.md)
- [Extraction Contract](01-extraction-contract.md)
- [Testing and CI](00-testing.md)
