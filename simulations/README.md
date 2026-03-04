# Simulations

## Purpose
Synthetic traffic and artifact simulation tooling used for demos and runtime proof workflows.

## Key Entrypoints
- `simulations/traffic/`
- `simulations/artifacts/`
- `simulations/tests/`

## Run/Test
```bash
uv run python -m pytest -q simulations/tests
```

## Dependencies
- Exercises running server/policy deployments and writes evidence under `traffic_out/`.

## Deep Links
- [`/docs/02-project-demos.md`](../docs/02-project-demos.md)
