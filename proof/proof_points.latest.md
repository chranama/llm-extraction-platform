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

## Proof 4: Async extract jobs
- Claim: extract requests can be queued and executed by a separate worker process with durable job state.
- Command: `python proof/generate_canonical_manifest.py`
- Artifacts:
  - `proof/artifacts/phase6_extract_async/async_submit_response.json`
  - `proof/artifacts/phase6_extract_async/async_job_initial.json`
  - `proof/artifacts/phase6_extract_async/async_job_final.json`
  - `proof/artifacts/phase6_extract_async/async_worker_log.txt`
  - `proof/artifacts/phase6_extract_async/async_job_summary.json`
- Validation signal: submit returns `202`, worker log includes the queued job id, and final status is `succeeded` with a valid result object.

## Proof 5: Traceable request inspection
- Claim: sync and async extract flows can be inspected as ordered per-request timelines, including async cross-process lineage.
- Command: `python proof/generate_canonical_manifest.py`
- Artifacts:
  - `proof/artifacts/phase7_trace_inspection/async_submit_response.json`
  - `proof/artifacts/phase7_trace_inspection/async_trace_detail.json`
  - `proof/artifacts/phase7_trace_inspection/async_trace_timeline.md`
  - `proof/artifacts/phase7_trace_inspection/sync_extract_response.json`
  - `proof/artifacts/phase7_trace_inspection/sync_trace_detail.json`
  - `proof/artifacts/phase7_trace_inspection/trace_summary.json`
- Validation signal: sync and async trace detail endpoints return coherent ordered events, and the async trace includes submission, worker, and status-poll lineage.
