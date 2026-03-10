# Proof System

This directory contains the canonical, latest-only proof bundle for recruiter and reviewer verification.

## Files
- `evidence_contract.schema.json`: shared evidence contract
- `evidence_manifest.latest.json`: machine-readable latest manifest
- `proof_points.latest.md`: human-readable proof summary
- `generate_canonical_manifest.py`: refreshes canonical manifest metadata
- `validate_evidence_manifest.py`: strict validator (schema-lite + artifact checks)

## Regenerate

```bash
python proof/generate_canonical_manifest.py
```

## Validate

```bash
python proof/validate_evidence_manifest.py
```
