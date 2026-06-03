# CLI

## Purpose
CLI entrypoints and command composition for local operations and developer workflows.

## Key Entrypoints
- `cli/commands/`
- `cli/utils/`
- `cli/tests/`

## Run/Test
```bash
uv run --project cli pytest -q
```

## Dependencies
- Uses project configuration and compose wrappers in `deploy/`.

## Deep Links
- [`/docs/operations.md`](../docs/operations.md)
