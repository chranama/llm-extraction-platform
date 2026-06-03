# Artifacts

The repository keeps stable runtime evidence under `proof/`.

## Main Files

- `proof/evidence_contract.schema.json`: schema for the evidence manifest.
- `proof/evidence_manifest.latest.json`: current machine-readable manifest.
- `proof/proof_points.latest.md`: human-readable summary of generated evidence.
- `proof/artifacts/`: saved outputs from the current runtime evidence bundle.
- `proof/validate_evidence_manifest.py`: validates the current bundle.
- `proof/generate_canonical_manifest.py`: regenerates the bundle.

## Validate Current Artifacts

```bash
python proof/validate_evidence_manifest.py
```

This is the lighter command. It checks the manifest contract and expected file
presence/signals for the current saved bundle.

## Regenerate Artifacts

```bash
python proof/generate_canonical_manifest.py
```

This command runs live local workflows and refreshes the manifest and summary.
It may require Docker, `kind`, Redis, Postgres, and the configured local model
profile.

## Artifact Groups

- `phase3_generate/`: generate policy clamp control and clamp manifests.
- `phase41_extract_gate/`: PASS/FAIL extraction gate outputs.
- `phase5_k8s_kind/`: local Kubernetes smoke output and rendered overlays.
- `phase6_extract_async/`: async job submission, worker, and final status outputs.
- `phase7_trace_inspection/`: sync and async trace inspection outputs.

## Scope Boundaries

The artifacts show local, reproducible behavior for selected runtime paths. They
do not show production-scale GPU scheduling, autoscaling under real traffic,
external tracing compliance, or horizontal worker orchestration.

Generated artifacts should stay close to the behavior they support. If a new
artifact does not clarify current runtime behavior, it should not be added to the
stable saved bundle.
