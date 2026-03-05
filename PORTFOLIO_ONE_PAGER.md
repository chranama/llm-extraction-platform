# LLM Extraction Platform — Portfolio One Pager

## What this project demonstrates

An end-to-end AI engineering system that combines:
- LLM serving,
- structured extraction contracts,
- offline eval-driven capability decisions,
- policy-based runtime controls,
- production-style testing and CI.

## Two flagship proofs

1. Generate Clamp
- Problem: latency spikes can degrade generation behavior.
- Control: policy runtime decision clamps generation token cap when SLO thresholds are exceeded.
- Proof: runtime policy artifact + response-level clamp fields.

2. Extract Gate
- Problem: not all models are safe/reliable for schema-constrained extraction.
- Control: onboarding artifacts toggle extract capability; server enforces gating.
- Proof: PASS/FAIL artifact split and endpoint behavior divergence.

## Core engineering skills signaled

- LLM systems integration
- backend/API contract design
- eval/policy lifecycle design
- test infrastructure and CI hardening
- deployment profile and operability discipline

## Where to review quickly

- Overview: [`README.md`](README.md)
- Demos: [`docs/02-project-demos.md`](docs/02-project-demos.md)
- Testing/CI: [`docs/00-testing.md`](docs/00-testing.md)
- Deployment behavior: [`docs/03-deployment-modes.md`](docs/03-deployment-modes.md)
- Model decision rationale: [`docs/10-model-decision-memo.md`](docs/10-model-decision-memo.md)
