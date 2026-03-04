# 06) Eval Methodology

This document defines how evaluation artifacts are used in this project.

## Goal

Produce reproducible evidence that drives onboarding/policy decisions, not just ad-hoc scores.

## Inputs

- dataset fixtures and prompts from eval pipelines
- model/profile under evaluation
- scenario metadata (run id, profile, threshold profile)

## Outputs

- eval run artifacts (summary + row-level data)
- pointers consumed by onboarding/policy flows
- pass/fail semantics used to patch model capability state

## How eval affects runtime

- Eval does not directly change server state.
- Onboarding consumes eval artifacts and writes patched model artifacts.
- Server behavior changes only when launched with those artifacts (`MODELS_YAML`).

## PASS/FAIL framing

For extract-gate demo:
- PASS artifact: target model extract capability enabled
- FAIL artifact: target model extract capability disabled
- Invariant: only extract capability differs for target model between PASS/FAIL artifacts

## Quality axes

Evaluate completeness on these axes:
- contract correctness
- error-path handling
- profile/deployment parity
- determinism/reproducibility
- artifact integrity and traceability

## Related docs

- [02-project-demos.md](02-project-demos.md)
- [04-architecture-deep-dive.md](04-architecture-deep-dive.md)
