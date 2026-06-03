# Runbook

This guide covers the routine local operator flow: prepare the environment,
start the system, verify it, inspect it while running, and shut it down.

Use [`operations.md`](operations.md) for runtime concepts and failure modes. Use
this file when you want the commands in order.

## Default Local Path

The default local path runs the API server, Postgres, and Redis through Docker
Compose using the repo-level `llmctl` command.

Use a local env override file for secrets and machine-specific model settings.
The examples below use `.env.docker`; replace it with your local file when
needed.

## Preflight

Install the repo-level CLI and development dependencies:

```bash
uv sync --extra dev
```

Confirm Docker is available:

```bash
docker info
```

Validate the Compose render before starting services:

```bash
uv run llmctl --project-name llmep --env-override-file .env.docker compose config --profiles infra server
```

## Start

Start the CPU Docker runtime and apply database migrations:

```bash
uv run llmctl --project-name llmep --env-override-file .env.docker dev dev-cpu
```

This starts the `infra` and `server` Compose profiles, builds the server image
when needed, and runs Alembic migrations inside the server container.

To start the UI after the API is running:

```bash
uv run llmctl --project-name llmep --env-override-file .env.docker compose up --profiles ui -d --remove-orphans
```

To start Prometheus and Grafana:

```bash
uv run llmctl --project-name llmep --env-override-file .env.docker compose up --profiles obs -d --remove-orphans
```

## Verify

Check running services:

```bash
uv run llmctl --project-name llmep --env-override-file .env.docker compose ps
```

Check liveness and readiness:

```bash
curl -fsS http://localhost:8000/healthz
curl -fsS http://localhost:8000/readyz
curl -fsS http://localhost:8000/v1/models/status
```

Run the Compose doctor:

```bash
uv run llmctl --project-name llmep --env-override-file .env.docker dev doctor
```

If an API key is configured, run a generation smoke check:

```bash
export API_KEY="<local-api-key>"
curl -fsS -X POST "http://localhost:8000/v1/generate" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ${API_KEY}" \
  --data '{"prompt":"smoke test","max_new_tokens":16,"temperature":0.2}'
```

## Observe

Follow service logs:

```bash
uv run llmctl --project-name llmep --env-override-file .env.docker compose logs --tail 200
```

Use `Ctrl-C` to stop following logs. This does not stop the running containers.

Useful runtime surfaces:

- API: `http://localhost:8000`
- UI: `http://localhost:5173`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`

## Stop

Stop the Compose project while preserving named volumes:

```bash
uv run llmctl --project-name llmep --env-override-file .env.docker compose down
```

Stop and remove named volumes when you need a clean database/cache state:

```bash
uv run llmctl --project-name llmep --env-override-file .env.docker compose down --volumes
```

## Kubernetes Local Path

The Kubernetes path is for local `kind` checks. It is separate from the default
Docker Compose runtime.

Start the local cluster and load the server image:

```bash
uv run llmctl k8s kind-up
uv run llmctl k8s kind-build-server
```

Apply the local generate-only overlay and wait for rollout:

```bash
uv run llmctl k8s apply-local-generate-only
uv run llmctl k8s wait
```

Inspect status and logs:

```bash
uv run llmctl k8s status
uv run llmctl k8s logs-api
```

Shut down the Kubernetes path:

```bash
uv run llmctl k8s delete-local-generate-only
uv run llmctl k8s kind-down
```

## Common Recovery Steps

If readiness fails, check the `readyz` response first. It reports DB, Redis,
model, policy, and deployment state.

If the server cannot reach Postgres or Redis, restart the default Compose path:

```bash
uv run llmctl --project-name llmep --env-override-file .env.docker compose down
uv run llmctl --project-name llmep --env-override-file .env.docker dev dev-cpu
```

If model readiness fails, check the selected model profile and any local model
paths in the env override file.

If policy-dependent routes are blocked unexpectedly, check
`policy_out/latest.json` and reload policy through the admin surface after
regenerating the policy artifact.
