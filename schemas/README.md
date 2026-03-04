# Schemas

## Purpose
JSON schema specification files used by extraction and related contract validation.

## Key Entrypoints
- `schemas/model_output/`
- `schemas/internal/`
- `schemas/tests/`

## Run/Test
```bash
uv run --project server --extra test pytest -q tests/integration/test_schema_registry_integration.py
```

## Dependencies
- Validation/type enforcement lives in `contracts/`.

## Deep Links
- [`/docs/01-extraction-contract.md`](../docs/01-extraction-contract.md)
