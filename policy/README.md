# Policy

## Purpose
Policy decision engine for runtime controls and onboarding decisions derived from eval/SLO artifacts.

## Key Entrypoints
- `policy/src/llm_policy/`
- `policy/tests/`
- `policy/src/llm_policy/thresholds/`

## Run/Test
```bash
uv run --project policy --extra test pytest -q
```

## Dependencies
- Consumes eval artifacts and writes decisions consumed by `server/` admin reload.

## Deep Links
- [`/docs/02-project-demos.md`](../docs/02-project-demos.md)
