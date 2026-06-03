# Scope

This repository is a local, inspectable backend system for LLM-backed product
workflows. It is intentionally bounded.

## In Scope

- Generate and schema-constrained extraction APIs.
- Runtime model capability checks.
- Policy-driven runtime controls.
- Sync and async extraction paths.
- Durable async job state.
- Structured errors and readiness behavior.
- Local Docker, compose, and Kubernetes-oriented deployment assets.
- CI-backed unit, integration, UI, and repo-level tests.
- Runtime evidence artifacts for selected behavior paths.

## Out Of Scope

- Production-scale GPU scheduling.
- Autoscaling under real traffic.
- High-availability operation.
- Production incident response.
- External distributed tracing compliance.
- Production secret-management hardening.
- Real customer traffic or service-level commitments.

## Current Tradeoff

The repository favors inspectability over minimal surface area. It includes
multiple subsystems because the target behavior crosses API serving, evaluation,
policy, deployment, and runtime evidence. The documentation and tests should make
those boundaries easy to inspect without presenting the project as a
production-operated service.

## Historical Documentation

Older documentation lives under [`archive/docs/`](../archive/docs/). It is kept
for project history and should not be treated as current implementation guidance.
