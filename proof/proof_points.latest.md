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

## Proof 3: Kubernetes kind deployment
- Claim: a local kind cluster runs the generate-only service successfully, while the production overlay renders as a production-oriented scaffold.
- Command: `python proof/generate_canonical_manifest.py`
- Artifacts:
  - `proof/artifacts/phase5_k8s_kind/kind_smoke_summary.json`
  - `proof/artifacts/phase5_k8s_kind/kubectl_get_pods.txt`
  - `proof/artifacts/phase5_k8s_kind/kubectl_get_svc.txt`
  - `proof/artifacts/phase5_k8s_kind/server_rollout_status.txt`
  - `proof/artifacts/phase5_k8s_kind/k8s_smoke.log`
  - `proof/artifacts/phase5_k8s_kind/kustomize_local_generate_only.yaml`
  - `proof/artifacts/phase5_k8s_kind/kustomize_prod_gpu_full.yaml`
- Validation signal: rollout passes, `/healthz` and generate smoke pass, `/v1/extract` is blocked, and both overlays render successfully.
