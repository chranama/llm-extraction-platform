# Config

## Purpose
Runtime profiles and environment-oriented configuration for server, models, and UI wiring.

## Key Entrypoints
- `config/server.yaml`
- `config/models.yaml`
- `config/ui.json`

## Run/Test
```bash
uv run --project server pytest -q tests/unit/test_llm_config_helpers_unit.py
```

## Dependencies
- Consumed by `server/`, `policy/`, `eval/`, and compose profiles.

## Deep Links
- [`/docs/03-deployment-modes.md`](../docs/03-deployment-modes.md)
