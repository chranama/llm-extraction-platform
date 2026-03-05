# 10) Model Decision Memo (Example with Concrete Evidence)

This memo captures a concrete decision for extract capability assignment using current repo artifacts.

## Context

- Candidate model: `sshleifer/tiny-gpt2`
- Use case: schema-constrained extraction in `host-transformers` profile
- Decision scope: onboarding capability assignment (`extract` true/false)

## Inputs reviewed

- Eval pass artifact: `results/extract/demo_extract_host_transformers_pass/`
- Eval fail artifact: `results/extract/demo_extract_host_transformers_fail/`
- Runtime proof run: `traffic_out/phase41_20260304T230327Z/`
- Evidence manifest: `traffic_out/phase41_20260304T230327Z/evidence_manifest.json`

## Decision

- PASS onboarding artifact keeps `extract=true`.
- FAIL onboarding artifact sets `extract=false`.

This is a reliability-first decision: block extract on fail artifact rather than allow unstable behavior.

## Rationale

- Runtime manifest confirms capability flip is enforced:
  - host pass capability: `true`
  - host fail capability: `false`
- Fail path probe returns `capability_disabled`, which is the intended guardrail contract.
- Pass path is not capability-blocked; observed failure there is model runtime (`model_load_failed`), not capability gating.

## Consequences

Positive:
- Clear capability contract at runtime.
- Reduced risk of unsupported extraction behavior.

Negative:
- Lower extract availability for weaker models.
- Additional onboarding/eval burden to enable more models safely.

## Evidence pointers

- Manifest (extract gate): `traffic_out/phase41_20260304T230327Z/evidence_manifest.json`
- PASS runtime: `traffic_out/phase41_20260304T230327Z/host_pass_runtime.json`
- FAIL runtime: `traffic_out/phase41_20260304T230327Z/host_fail_runtime.json`
- PASS extract probe: `traffic_out/phase41_20260304T230327Z/host_pass_extract.json`
- FAIL extract probe: `traffic_out/phase41_20260304T230327Z/host_fail_extract.json`

## Follow-up

Before changing this decision:
- rerun eval with updated model/profile,
- compare against prior manifests,
- rerun runtime proof to ensure capability semantics remain deterministic.
