# LLM Extraction Platform

Backend service for LLM-backed generate and structured extraction workflows.

The system exposes explicit API contracts around model outputs, applies runtime
policy and capability gates, and includes tests plus runtime evidence for sync
and async extraction behavior.

## What It Does

- Serves `/v1/generate` for text generation with runtime controls.
- Serves `/v1/extract` for schema-constrained structured extraction.
- Serves `/v1/extract/jobs` for queued extraction with durable job state.
- Exposes schema, model, health, readiness, admin, and trace inspection surfaces.
- Connects offline evaluation and policy artifacts to runtime capability decisions.

## System Boundaries

- `server/`: FastAPI runtime service for generate, extract, admin, health, readiness, and traces.
- `policy/`: policy engine for model capability and runtime control decisions.
- `eval/`: evaluation jobs and scoring workflows used by policy inputs.
- `contracts/` and `schemas/`: shared artifact models and extraction schemas.
- `integrations/`: repo-level tests for cross-service workflows.
- `deploy/`: compose, Docker, Kubernetes, observability, and proxy assets.
- `proof/`: generated runtime evidence, validation scripts, and stable artifacts.
- `ui/`: frontend surface for operating and inspecting the service.

## Commands

Install repo-level tooling:

```bash
uv sync --extra dev
```

Run representative repo tests:

```bash
uv run python -m pytest -q cli/tests config/tests contracts/tests schemas/tests tools/tests
```

Run service tests:

```bash
cd server
uv sync --extra test
uv run python -m pytest -q tests/unit
uv run python -m pytest -q tests/integration
```

Run policy tests:

```bash
uv run --project policy --extra test pytest -q
```

Run eval tests:

```bash
uv run --project eval --extra test pytest -q
```

Validate the current runtime evidence bundle:

```bash
python proof/validate_evidence_manifest.py
```

Regenerate the runtime evidence bundle:

```bash
python proof/generate_canonical_manifest.py
```

Regeneration runs live local workflows and may require Docker, Kubernetes
`kind`, Redis, Postgres, and the configured local model profile.

## Documentation

- [Architecture](docs/architecture.md)
- [API](docs/api.md)
- [Testing](docs/testing.md)
- [Operations](docs/operations.md)
- [Runbook](docs/runbook.md)
- [Artifacts](docs/artifacts.md)
- [Scope](docs/scope.md)

Older documentation has been archived under [`archive/docs/`](archive/docs/).

## Current Scope

This repository shows a local, inspectable backend system for LLM-backed product
workflows. It includes API contracts, runtime policy behavior, local deployment
assets, test coverage, and generated artifacts.

It does not claim production-scale GPU scheduling, autoscaling under real
traffic, external distributed tracing compliance, or high-availability operation.

## License

MIT License. See [`LICENSE`](LICENSE).
