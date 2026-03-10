# Canonical Proof Points (Latest)

## Proof 1: Generate Clamp
- Claim: runtime policy clamp behavior is reproducibly evidenced.
- Command: `python proof/generate_canonical_manifest.py`
- Artifacts:
  - `traffic_out/phase3_generate_20260304/evidence_manifest_control.json`
  - `traffic_out/phase3_generate_20260304/evidence_manifest_clamp.json`
- Validation signal: both control and clamp manifests exist with divergent expected outcomes.

## Proof 2: Extract Gate PASS/FAIL
- Claim: offline onboarding artifacts drive runtime extract capability enforcement.
- Command: `python proof/generate_canonical_manifest.py`
- Artifacts:
  - `traffic_out/phase41_20260304T230327Z/evidence_manifest.json`
  - `traffic_out/phase41_20260304T230327Z/host_pass_runtime.json`
  - `traffic_out/phase41_20260304T230327Z/host_fail_runtime.json`
- Validation signal: PASS and FAIL runtime outputs differ according to gating expectation.
