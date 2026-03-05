# Scripts

## Purpose
Operational wrappers for reproducible demo and validation flows.

## Key Entrypoints
- `scripts/demo_extract_gate/run_extract_gate_matrix.sh`
- `scripts/demo_extract_gate/run_host_transformers.sh`
- `scripts/demo_extract_gate/run_docker_llama.sh`
- `scripts/demo_extract_gate/write_evidence_manifest.py`
- `scripts/demo_generate_clamp/write_evidence_manifest.py`
- `scripts/ci_smoke_matrix.sh`

## Run/Test
```bash
scripts/demo_extract_gate/run_host_transformers.sh
python3 scripts/demo_generate_clamp/write_evidence_manifest.py --help
bash scripts/ci_smoke_matrix.sh
RUN_INTEGRATION=1 bash scripts/ci_smoke_matrix.sh
```

## Dependencies
- Uses compose profiles in `deploy/compose/` and tools in `simulations/`.

## Deep Links
- [`/docs/02-project-demos.md`](../docs/02-project-demos.md)
- [`/docs/09-ci-hardening.md`](../docs/09-ci-hardening.md)
