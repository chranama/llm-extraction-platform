# LLM Extraction Platform

Backend service for LLM-backed generate and structured extraction workflows.

The system exposes explicit API contracts around model outputs, applies runtime
policy and capability gates, and includes tests plus runtime evidence for sync
and async extraction behavior.

Text generation asks a model to produce new text from a prompt. Structured
extraction asks a model to read source text and return fields that match a
declared schema, such as receipt totals or invoice metadata.

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

## Run Locally

The local runbook provides the step-by-step guide for starting, verifying,
inspecting, and shutting down the platform:

- [Runbook](docs/runbook.md)

It covers the supported reviewer smoke, Compose extract, external model,
joint inference-gateway, Kubernetes smoke, policy/eval linkage, admin/trace,
ops-surface, and evidence-validation paths. The promoted joint local path runs
LLMEP and the gateway together in `kind` with a live CPU-only `llama.cpp` model
server.

## Documentation

- [Architecture](docs/architecture.md)
- [API](docs/api.md)
- [Testing](docs/testing.md)
- [Operations](docs/operations.md)
- [Runbook](docs/runbook.md)
- [Runtime Setup](docs/runtime-setup.md)
- [Inference Gateway Integration](docs/inference-gateway-integration.md)
- [API And Model Runtime Evolution](docs/decisions/api-model-runtime-evolution.md)
- [Artifacts](docs/artifacts.md)
- [Scope](docs/scope.md)

Older documentation has been archived under [`archive/docs/`](archive/docs/).

## Current Scope

This repository shows a local, inspectable backend system for LLM-backed product
workflows. It includes API contracts, runtime policy behavior, local deployment
assets, test coverage, and generated artifacts.

It does not claim production-scale GPU scheduling, autoscaling under real
traffic, full benchmark evaluation coverage, external distributed tracing
compliance, or high-availability operation.

## License

MIT License. See [`LICENSE`](LICENSE).
