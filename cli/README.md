# CLI

## Purpose
CLI entrypoints for the repository's promoted happy paths.

`llmctl` is not intended to replace `docker compose`, `kubectl`, `curl`, or the
proof scripts. It launches curated workflows, renders deterministic Compose
environment files, applies migrations when needed, and runs the core smoke
checks for supported runtime targets.

## Key Entrypoints
- `llmctl preflight <target>`
- `llmctl smoke`
- `llmctl compose-extract`
- `llmctl external-model`
- `llmctl kind-smoke`
- `llmctl evidence`
- `llmctl doctor`
- `llmctl stop`
- `cli/commands/`
- `cli/utils/`
- `cli/tests/`

## Preflight
Use `preflight` before starting a promoted runtime target. It validates local
prerequisites, rendered Compose wiring, required env values, model/policy/schema
files, and port availability without starting containers or mutating runtime
state.

```bash
uv run llmctl --env-override-file .env.docker preflight smoke
uv run llmctl --env-override-file .env.docker preflight compose-extract
uv run llmctl preflight evidence --json
```

## Run/Test
```bash
uv run --project cli pytest -q
```

## Dependencies
- Uses project configuration and compose wrappers in `deploy/`.
- Keeps lower-level `compose`, `dev`, and `k8s` command groups available for
  advanced development workflows.

## Deep Links
- [`/docs/operations.md`](../docs/operations.md)
