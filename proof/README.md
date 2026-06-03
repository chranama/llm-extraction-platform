# Runtime Evidence

This directory contains the latest saved runtime evidence bundle and validation scripts.

## Files
- `evidence_contract.schema.json`: shared evidence contract
- `evidence_manifest.latest.json`: machine-readable latest manifest
- `proof_points.latest.md`: human-readable evidence summary
- `artifacts/`: tracked canonical latest-only proof bundle used by CI validation
- `generate_canonical_manifest.py`: canonical evidence entrypoint; regenerates the Kubernetes and async extraction artifacts and refreshes the manifest/summary
- `generate_k8s_kind_proof.py`: live `kind` deployment evidence helper
- `generate_async_extract_proof.py`: local async extraction evidence helper
- `generate_trace_inspection_proof.py`: sync and async request-trace evidence helper
- `validate_evidence_manifest.py`: strict validator (schema-lite + artifact checks)
- `fixtures/models.gateway-proof.yaml`: deterministic fake-model profile for live gateway-backed extract proof

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
- refreshes `evidence_manifest.latest.json`
- refreshes `proof_points.latest.md`

## Validate

```bash
python proof/validate_evidence_manifest.py
```

Local `kind` evidence shows runnable Kubernetes deployment. Production overlay render shows scaffold readiness only. This evidence does not claim real GPU scheduling or production-scale operation.

Async extraction evidence shows queue-backed job submission, separate worker execution, and durable job-state polling. It does not claim production-scale queue operations, retries, or horizontal worker orchestration.

Trace inspection evidence shows ordered per-request timelines for sync and async extract flows, including async cross-process lineage. It does not claim distributed tracing standards compliance or external telemetry export.
