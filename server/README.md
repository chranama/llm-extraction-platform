# Server

## Purpose
Primary runtime API service exposing generate/extract/admin/health endpoints.

## Key Entrypoints
- `server/src/llm_server/`
- `server/tests/unit/`
- `server/tests/integration/`

## Run/Test
```bash
cd server
uv sync --extra test
uv run python -m pytest -q tests/unit
uv run python -m pytest -q tests/integration
```

## Dependencies
- Reads config from `config/`; uses contracts from `contracts/` and schemas from `schemas/`.

## Deep Links
- [`/docs/01-extraction-contract.md`](../docs/01-extraction-contract.md)
- [`/docs/03-deployment-modes.md`](../docs/03-deployment-modes.md)
