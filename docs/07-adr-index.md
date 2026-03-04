# 07) ADR Index

Architecture Decision Records (ADRs) capture the “why” behind key design choices.

## Current decisions (seed set)

1. Schema spec vs contract enforcement split
- Decision: keep schema files in `schemas/` and Python enforcement in `contracts/`.
- Rationale: explicit boundary between data contracts and implementation details.

2. Artifact-driven runtime behavior
- Decision: use offline artifacts (`MODELS_YAML`, policy decision files) to control runtime behavior.
- Rationale: reproducibility, auditability, and deterministic demo flows.

3. Layered test strategy
- Decision: package-level unit/integration + repo-level integration lanes.
- Rationale: isolate failures while preserving end-to-end confidence.

4. Deployment profile composition
- Decision: profile composition via `llmctl` + compose defaults.
- Rationale: explicit, reusable deployment wiring across host/docker/llama variants.

## ADR format for future additions

When adding an ADR, use:
- Context
- Decision
- Consequences (positive/negative)
- Alternatives considered
- Date and owner

## Related docs

- [04-architecture-deep-dive.md](04-architecture-deep-dive.md)
- [03-deployment-modes.md](03-deployment-modes.md)
