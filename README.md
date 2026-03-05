# LLM Extraction Platform

Portfolio project demonstrating production-style AI engineering: serving LLM features, enforcing extraction policy, validating structured outputs, and operating with test-first reliability across multiple deployment modes.

## Why This Project Exists

I built this repository to demonstrate skills relevant to an entry-level AI Engineer role:
- building and operating LLM-backed product features,
- designing backend APIs with clear contracts,
- evaluating model behavior and enforcing policy decisions,
- shipping reliable systems with CI, integration tests, and operational diagnostics.

## How To Review This Repo

Recommended review flow (5-10 minutes):
1. Read the two demo proof cards below.
2. Run one demo command and inspect the produced evidence manifest.
3. Open the docs index for deeper technical references.

Deep index: [`docs/README.md`](docs/README.md)

## Demo Proof Cards

### Demo 1: Generate Clamp
- Risk: tail-latency spikes can degrade UX/cost.
- Control: policy runtime decision applies `generate_max_new_tokens_cap` when SLO threshold is exceeded.
- One-command repro (scripted evidence writer):
  - `scripts/demo_generate_clamp/write_evidence_manifest.py --slo slo_out/generate/latest.json --policy policy_out/latest.json --out traffic_out/generate_clamp_latest/evidence_manifest.json`
- Evidence artifact:
  - `traffic_out/<run>/evidence_manifest.json` with baseline/clamp status and cap proof.

### Demo 2: Extract Gate
- Risk: unsupported models silently producing invalid extraction behavior.
- Control: offline onboarding artifacts toggle extract capability in `models.patched.*.yaml`; server enforces capability at runtime.
- One-command repro:
  - `scripts/demo_extract_gate/run_host_transformers.sh`
- Evidence artifact:
  - `traffic_out/<run>/evidence_manifest.json` with PASS/FAIL capability + endpoint behavior.

## Skills Demonstrated

### 1) LLM Systems Engineering
- Generate and schema-constrained extract capabilities with explicit gating.
- Evidence: [`server/README.md`](server/README.md), [`policy/README.md`](policy/README.md), [`schemas/README.md`](schemas/README.md)

### 2) Backend/API Design
- Versioned endpoints, auth/limits, health/readiness, and structured error contracts.
- Evidence: [`docs/01-extraction-contract.md`](docs/01-extraction-contract.md), [`server/README.md`](server/README.md)

### 3) Evaluation and Policy Lifecycle
- Offline evaluation artifacts feed policy/onboarding decisions that affect runtime behavior.
- Evidence: [`eval/README.md`](eval/README.md), [`policy/README.md`](policy/README.md), [`docs/02-project-demos.md`](docs/02-project-demos.md)

### 4) Reliability and Test Infrastructure
- Unit/integration/live coverage split across services with CI execution.
- Evidence: [`docs/00-testing.md`](docs/00-testing.md), [`.github/workflows/ci.yml`](.github/workflows/ci.yml)

### 5) Deployment and Operability
- Host and container deployment modes with reproducible scripts and diagnostics capture.
- Evidence: [`deploy/README.md`](deploy/README.md), [`scripts/README.md`](scripts/README.md), [`docs/03-deployment-modes.md`](docs/03-deployment-modes.md)

## Architecture At A Glance

- `server/`: runtime API for generate/extract/admin health and readiness.
- `policy/`: policy decision engine and onboarding logic.
- `eval/`: evaluation jobs and reporting pipeline.
- `contracts/` + `schemas/`: validation layer and schema specs.
- `integrations/`: repo-level end-to-end and live integration scenarios.
- `ui/`: frontend surfaces for interacting with the system.

## 10-Minute Reviewer Path

1. Read architecture and scope:
- [`docs/03-deployment-modes.md`](docs/03-deployment-modes.md)
- [`docs/01-extraction-contract.md`](docs/01-extraction-contract.md)

2. Run extract-gate demo validation (fast operational proof):
```bash
scripts/demo_extract_gate/run_host_transformers.sh
```

3. Inspect produced evidence:
- `traffic_out/<latest run>/evidence_manifest.json`
- `traffic_out/<latest run>/host_pass_runtime.json`
- `traffic_out/<latest run>/host_fail_runtime.json`

4. Review CI and tests:
- [`.github/workflows/ci.yml`](.github/workflows/ci.yml)
- [`docs/00-testing.md`](docs/00-testing.md)

## Testing And CI Signals

- Service-level unit and integration suites in `server/`, `policy/`, `eval/`, `ui/`.
- Repo-level integration suites in `integrations/`.
- CI matrix in [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

## Repository Map

- [`cli/README.md`](cli/README.md)
- [`config/README.md`](config/README.md)
- [`contracts/README.md`](contracts/README.md)
- [`deploy/README.md`](deploy/README.md)
- [`docs/README.md`](docs/README.md)
- [`eval/README.md`](eval/README.md)
- [`integrations/README.md`](integrations/README.md)
- [`policy/README.md`](policy/README.md)
- [`schemas/README.md`](schemas/README.md)
- [`scripts/README.md`](scripts/README.md)
- [`server/README.md`](server/README.md)
- [`simulations/README.md`](simulations/README.md)
- [`tools/README.md`](tools/README.md)
- [`ui/README.md`](ui/README.md)

## Future Improvements

### Near-term
- Broaden integration smoke coverage for deployment/profile variants.
- Keep docs CI checks strict for links and stale command patterns.
- Expand reviewer quickstart with expected outputs/screenshots.
- Improve architecture visual clarity.

### Mid-term
- Deepen evaluation rigor (error taxonomy, regressions, confidence reporting).
- Calibrate policy thresholds from traffic/eval statistics.
- Add SLO dashboards and incident-response walkthrough docs.
- Formalize dataset/prompt versioning and lineage docs.

### Long-term
- Add additional serving backends (for example vLLM/TGI) with benchmark docs.
- Build model lifecycle governance docs (onboarding/offboarding/rollback).
- Add cost-performance optimization framework and reporting.
- Add security hardening docs (threat model, abuse tests, supply-chain controls).

## License

MIT License. See [`LICENSE`](LICENSE).
