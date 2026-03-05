# 08) Demo Evidence Index

This page maps each demo to expected evidence artifacts.

## Demo 1: Generate Clamp

Reference: [02-project-demos.md](02-project-demos.md)

Expected evidence:
- SLO artifact: `slo_out/generate/latest.json`
- Policy artifact: `policy_out/latest.json`
- Evidence manifest: `traffic_out/<run>/evidence_manifest.json`
- Runtime reload response with non-null `generate_max_new_tokens_cap`
- Generate response showing clamp fields:
  - `requested_max_new_tokens`
  - `effective_max_new_tokens`
  - `clamped`

## Demo 2: Extract Gate

Reference: [02-project-demos.md](02-project-demos.md)

Expected evidence:
- PASS/FAIL patched model artifacts under `config/models.patched.*.yaml`
- Evidence manifest: `traffic_out/<run>/evidence_manifest.json`
- `/v1/models` response showing target model extract capability toggled
- PASS run extract success path
- FAIL run capability-blocked extract path
- Script diagnostics under `traffic_out/<run_tag>/diagnostics/`

## Reviewer checklist

- Can you trace cause -> artifact -> runtime behavior?
- Do PASS/FAIL runs differ only by intended control input?
- Is each claim backed by concrete file/log/API evidence?
