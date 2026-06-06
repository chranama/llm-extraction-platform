# Runtime Evidence

This directory contains the latest saved runtime evidence bundle and validation scripts.

## Files
- `evidence_contract.schema.json`: shared evidence contract
- `evidence_manifest.latest.json`: machine-readable latest manifest
- `proof_points.latest.md`: human-readable evidence summary
- `artifacts/`: tracked canonical latest-only proof bundle used by CI validation
- `generate_canonical_manifest.py`: canonical evidence entrypoint; regenerates the Kubernetes, async extraction, trace, policy/eval, and ops-surface artifacts and refreshes the manifest/summary
- `generate_k8s_kind_proof.py`: live `kind` deployment evidence helper
- `generate_async_extract_proof.py`: local async extraction evidence helper
- `generate_trace_inspection_proof.py`: sync and async request-trace evidence helper
- `generate_compose_llama_extract_proof.py`: live Compose extract helper using
  CPU-only containerized `llama.cpp`
- `generate_policy_eval_linkage_proof.py`: eval-to-policy and admin reload evidence helper
- `generate_ops_surface_proof.py`: API/UI/observability/proxy evidence helper
- `validate_evidence_manifest.py`: strict validator (schema-lite + artifact checks)
- `fixtures/models.gateway-proof.yaml`: deterministic fake-model profile for live gateway-backed extract proof

The joint LLMEP plus `inference-serving-gateway` workflow is started from
`tools/joint/inference_gateway_stack.sh` and writes optional artifacts under
`artifacts/joint_gateway/`.

## Regenerate

```bash
python proof/generate_canonical_manifest.py
```

This command now:
- deploys the local generate-only overlay to `kind`
- runs the Kubernetes smoke checks
- renders the local and production overlays
- runs the async extract job proof with a separate worker process
- runs the sync and async trace inspection proof
- runs the policy/eval linkage proof
- runs the UI/observability/proxy proof
- refreshes `evidence_manifest.latest.json`
- refreshes `proof_points.latest.md`

The phase 8 Compose llama proof is included in the manifest as saved evidence,
but it is generated separately because it requires a machine-specific GGUF model
file.

## Validate

```bash
python proof/validate_evidence_manifest.py
```

Local `kind` evidence shows runnable Kubernetes deployment. Production overlay render shows scaffold readiness only. This evidence does not claim real GPU scheduling or production-scale operation.

Async extraction evidence shows queue-backed job submission, separate worker execution, and durable job-state polling. It does not claim production-scale queue operations, retries, or horizontal worker orchestration.

Trace inspection evidence shows ordered per-request timelines for sync and async extract flows, including async cross-process lineage. It does not claim distributed tracing standards compliance or external telemetry export.

Compose llama extract evidence shows model-backed extraction with a lightweight
containerized `llama.cpp` backend. It does not claim accelerated inference,
production throughput, or cloud model-serving readiness.

Policy/eval linkage evidence shows that deterministic eval summaries produce
policy decisions and that admin reload changes runtime extract behavior. It does
not claim broad benchmark coverage or a full dataset evaluation run.

Ops-surface evidence shows local API, UI, Prometheus, and Grafana reachability
directly and through nginx. It also captures Prometheus scrape state and Grafana
dashboard population. It does not claim production ingress, TLS, identity
controls, or production observability operations.

Joint gateway evidence shows sync and async extraction through the companion
edge gateway with request identity, trace identity, metrics, admin traces, and
optional OTel export. It uses a deterministic fake model profile and does not
claim real-model quality or cloud deployment behavior.

Joint workflow artifact groups include:

- `joint_gateway/latest/`: deterministic compatibility proof
- `joint_gateway/observability_latest/`: deterministic proof with OTel evidence
- `joint_gateway/edge_controls_latest/`: gateway route/limit/error controls
- `joint_gateway/llama_extract_latest/`: real model-backed CPU llama.cpp extract
- `joint_gateway/containerized_latest/`: local all-container Compose stack
- `joint_gateway/containerized_llama_latest/`: local all-container Compose stack
  with CPU llama.cpp extract
- `joint_gateway/kind_smoke_latest/`: local Kubernetes-shaped smoke proof
