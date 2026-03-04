# Integrations

## Purpose
Repository-level integration suites validating cross-service behavior and live pathways.

## Key Entrypoints
- `integrations/test_e2e_loop/`
- `integrations/test_eval_job/`
- `integrations/test_policy_job/`
- `integrations/test_server_live/`

## Run/Test
```bash
cd integrations
uv sync --extra test
uv run pytest -q
```

## Dependencies
- Exercises compose deployments and interactions across `server/`, `eval/`, and `policy/`.

## Deep Links
- [`/docs/00-testing.md`](../docs/00-testing.md)
