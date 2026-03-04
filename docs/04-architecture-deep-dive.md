# 04) Architecture Deep Dive

This document explains how the major packages interact at runtime and through control-plane artifacts.

## System components

- `server/`: online API plane for generate/extract/admin endpoints.
- `policy/`: policy decision engine and onboarding logic.
- `eval/`: offline evaluation jobs that produce scored artifacts.
- `schemas/`: JSON schema specs.
- `contracts/`: Python validation/types used across services/jobs.
- `integrations/`: repo-level end-to-end suites.
- `ui/`: frontend client.

## Request path (extract)

1. Client calls `POST /v1/extract` in `server`.
2. Server resolves active model profile from `MODELS_YAML` + `MODELS_PROFILE`.
3. Server checks extract capability/policy gating.
4. LLM runtime generates output.
5. Output is parsed and schema-validated against `schemas/model_output/sroie_receipt_v1.json`.
6. Validated payload is returned; metrics/logs emitted.

## Control-plane artifact path

1. `eval/` produces run artifacts.
2. `policy/` onboarding consumes eval artifacts and writes patched model config artifacts.
3. `server/` is pointed at those artifacts through `MODELS_YAML`.
4. Runtime behavior changes deterministically based on artifact state.

## Policy decision path (generate clamp)

1. `server` exports runtime SLO snapshot (`runtime_generate_slo_v1`).
2. `policy runtime-decision` consumes SLO + threshold profile.
3. Policy artifact (`policy_decision_v2`) is written.
4. `server` admin reload applies decision; generate cap is enforced at request-time.

## Deployment shapes

- Host server + docker infra (`infra-host`).
- Docker server + docker infra (`infra`).
- Docker server + in-compose llama (`server-llama`).
- Docker server + host llama (`server-llama-host`).

See [03-deployment-modes.md](03-deployment-modes.md) for exact profile wiring.

## Design constraints

- Capability-aware behavior (extract may be disabled/blocked per model/profile).
- Schema spec and contract enforcement split is explicit (`schemas/` vs `contracts/`).
- Offline artifacts control runtime behavior; this is intentional for reproducibility.
- Tests are layered: package unit/integration + repo integration lanes.

## Related docs

- [00-testing.md](00-testing.md)
- [01-extraction-contract.md](01-extraction-contract.md)
- [02-project-demos.md](02-project-demos.md)
- [03-deployment-modes.md](03-deployment-modes.md)
