# Canonical Proof Points (Latest)

## Proof 1: Generate Clamp
- Claim: runtime policy clamp behavior is reproducibly evidenced.
- Command: `python proof/generate_canonical_manifest.py`
- Artifacts:
  - `proof/artifacts/phase3_generate/evidence_manifest_control.json`
  - `proof/artifacts/phase3_generate/evidence_manifest_clamp.json`
- Validation signal: both control and clamp manifests exist with divergent expected outcomes.

## Proof 2: Extract Gate PASS/FAIL
- Claim: offline onboarding artifacts drive runtime extract capability enforcement.
- Command: `python proof/generate_canonical_manifest.py`
- Artifacts:
  - `proof/artifacts/phase41_extract_gate/evidence_manifest.json`
  - `proof/artifacts/phase41_extract_gate/host_pass_runtime.json`
  - `proof/artifacts/phase41_extract_gate/host_fail_runtime.json`
- Validation signal: PASS and FAIL runtime outputs differ according to gating expectation.
