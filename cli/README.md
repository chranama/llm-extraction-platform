# CLI

## Purpose
CLI entrypoints for the repository's promoted happy paths.

`llmctl` is not intended to replace `docker compose`, `kubectl`, `curl`, or the
proof scripts. It launches curated workflows, renders deterministic Compose
environment files, applies migrations when needed, and runs the core smoke
checks for supported runtime targets.

## Key Entrypoints
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
