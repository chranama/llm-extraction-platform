# Eval

## Purpose
Evaluation jobs and scoring workflows used for model quality and policy/onboarding input.

## Key Entrypoints
- `eval/src/llm_eval/`
- `eval/tests/`

## Run/Test
```bash
uv run --project eval --extra test pytest -q
```

## Dependencies
- Produces artifacts consumed by `policy/` onboarding logic.

## Deep Links
- [`/docs/02-project-demos.md`](../docs/02-project-demos.md)
