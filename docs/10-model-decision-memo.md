# 10) Model Decision Memo (Example)

This memo template captures the decision rationale for accepting or rejecting a model/profile for extract workloads.

## Context

- Candidate model: `sshleifer/tiny-gpt2` (example profile case)
- Use case: schema-constrained extraction
- Decision scope: onboarding capability assignment (`extract` true/false)

## Inputs reviewed

- eval artifacts for pass/fail scenarios
- threshold profile and criteria
- runtime extract-gate proof outputs

## Decision

Example decision:
- Set `extract=false` for this candidate in FAIL onboarding path.

## Rationale

- The model failed reliability criteria for structured extract behavior under required constraints.
- Allowing extract would increase invalid-output risk and operational support burden.

## Consequences

Positive:
- Lower probability of invalid extract behavior in production-like paths.
- Clear capability contract at runtime.

Negative:
- Reduced model availability for extract workloads.
- Need stronger model or tuned profile for extract enablement.

## Evidence pointers

- Eval outputs: `results/extract/<run>/...`
- Onboarding artifacts: `config/models.patched.*.yaml`
- Runtime proof: `traffic_out/<run>/...`
- Evidence manifest: `traffic_out/<run>/evidence_manifest.json`

## Follow-up

Before revisiting this decision:
- rerun eval with updated model/profile,
- compare against prior artifact metrics,
- replay extract-gate runtime checks.
