# 06) Eval Methodology

This document defines how evaluation artifacts are used in this project and how threshold/policy decisions are justified.

## Goal

Produce reproducible evidence that drives onboarding/policy decisions, not ad-hoc scores.

## Inputs

- dataset fixtures and prompts from eval pipelines
- model/profile under evaluation
- scenario metadata (run id, profile, threshold profile)
- policy threshold profile used for decisioning

## Outputs

- eval run artifacts (summary + row-level data)
- pointers consumed by onboarding/policy workflows
- pass/fail semantics used to patch model capability state

## How eval affects runtime

- Eval does not directly change server state.
- Onboarding consumes eval artifacts and writes patched model artifacts.
- Server behavior changes only when launched with those artifacts (`MODELS_YAML`).

## PASS/FAIL framing

For extract-gate demo:
- PASS artifact: target model extract capability enabled.
- FAIL artifact: target model extract capability disabled.
- Invariant: only extract capability differs for target model between PASS/FAIL artifacts.

## Threshold rationale and tradeoffs

Thresholds are selected to balance two risks:
1. False allow: model is admitted for extract despite unstable structured-output behavior.
2. False block: model is blocked even though it is operationally acceptable.

Current policy is biased toward reliability for extract paths.

Tradeoff guidance:
- Tight thresholds reduce bad production outputs but may reduce model availability.
- Relaxed thresholds improve availability but increase schema/quality risk.

Threshold changes should be paired with:
- explicit before/after metric comparison,
- failure-case replay,
- updated policy/eval documentation.

## Quality axes

Evaluate completeness on these axes:
- contract correctness
- error-path handling
- profile/deployment parity
- determinism/reproducibility
- artifact integrity and traceability

## Evidence requirements for decision claims

Any model/onboarding decision claim should link to:
- eval summary artifact,
- threshold profile used,
- resulting onboarding or policy artifact,
- runtime proof (if decision affects runtime behavior).

## Related docs

- [02-project-demos.md](02-project-demos.md)
- [04-architecture-deep-dive.md](04-architecture-deep-dive.md)
- [10-model-decision-memo.md](10-model-decision-memo.md)
