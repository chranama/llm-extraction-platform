# Proof System

This directory contains the canonical, latest-only proof bundle for recruiter and reviewer verification.

## Files
- `evidence_contract.schema.json`: shared evidence contract
- `evidence_manifest.latest.json`: machine-readable latest manifest
- `proof_points.latest.md`: human-readable proof summary
- `artifacts/`: tracked canonical latest-only proof bundle used by CI validation
- `generate_canonical_manifest.py`: canonical proof entrypoint; regenerates the Kubernetes and async extraction proof artifacts and refreshes the manifest/summary
- `generate_k8s_kind_proof.py`: live `kind` deployment proof helper
- `generate_async_extract_proof.py`: local async extraction proof helper
- `validate_evidence_manifest.py`: strict validator (schema-lite + artifact checks)

## Regenerate

```bash
python proof/generate_canonical_manifest.py
```

This command now:
- deploys the local generate-only overlay to `kind`
- runs the Kubernetes smoke checks
- renders the local and production overlays
- runs the async extract job proof with a separate worker process
- refreshes `evidence_manifest.latest.json`
- refreshes `proof_points.latest.md`

## Validate

```bash
python proof/validate_evidence_manifest.py
```

Local `kind` proof demonstrates runnable Kubernetes deployment. Production overlay render demonstrates scaffold readiness only. This proof does not claim real GPU scheduling or production-scale operation.

Async extraction proof demonstrates queue-backed job submission, separate worker execution, and durable job-state polling. It does not claim production-scale queue operations, retries, or horizontal worker orchestration.
