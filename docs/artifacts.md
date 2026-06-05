# Artifacts

The repository keeps stable runtime evidence under `proof/`.

## Main Files

- `proof/evidence_contract.schema.json`: schema for the evidence manifest.
- `proof/evidence_manifest.latest.json`: current machine-readable manifest.
- `proof/proof_points.latest.md`: human-readable summary of generated evidence.
- `proof/artifacts/`: saved outputs from the current runtime evidence bundle.
- `proof/validate_evidence_manifest.py`: validates the current bundle.
- `proof/generate_canonical_manifest.py`: regenerates the bundle.
- `proof/generate_compose_llama_extract_proof.py`: runs the promoted Compose
  extract path with containerized `llama.cpp` and writes phase 8 artifacts.
- `proof/generate_policy_eval_linkage_proof.py`: runs the eval-to-policy
  linkage proof and writes phase 9 artifacts.
- `proof/generate_ops_surface_proof.py`: runs the API/UI/observability/proxy
  proof and writes phase 10 artifacts.

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
It may require Docker, `kind`, Redis, and Postgres. It does not regenerate the
phase 8 Compose llama proof, because that proof requires a machine-specific
GGUF model file.

## Generate Compose Llama Extract Artifacts

```bash
python proof/generate_compose_llama_extract_proof.py
```

This command requires `.env.docker` or `PHASE8_ENV_FILE` to point at a local
GGUF model. It starts the promoted Compose extract path with CPU-only
containerized `llama.cpp`, verifies generate, sync extract, and async extract,
captures logs, and writes `phase8_compose_llama_extract/`.

## Artifact Groups

- `phase3_generate/`: generate policy clamp control and clamp manifests.
- `phase41_extract_gate/`: PASS/FAIL extraction gate outputs.
- `phase5_k8s_kind/`: local Kubernetes smoke output and rendered overlays.
- `phase6_extract_async/`: async job submission, worker, and final status outputs.
- `phase7_trace_inspection/`: sync and async trace inspection outputs.
- `phase8_compose_llama_extract/`: real-model Compose extract output from
  containerized `llama.cpp` when generated locally.
- `phase9_policy_eval_linkage/`: eval summaries, generated policy decisions,
  admin policy reload responses, and allow/deny extract responses.
- `phase10_ops_surface/`: direct and proxied API, UI, Prometheus, Grafana, and
  Compose status outputs, including Prometheus scrape state and Grafana
  dashboard population.

## Scope Boundaries

The artifacts show local, reproducible behavior for selected runtime paths. They
do not show production-scale GPU scheduling, autoscaling under real traffic,
external tracing compliance, or horizontal worker orchestration.

`phase8_compose_llama_extract/` demonstrates CPU-only model-backed extraction
correctness. It does not demonstrate accelerated inference or production model
throughput.

`phase9_policy_eval_linkage/` demonstrates the control loop from eval summary to
policy artifact to runtime behavior with deterministic eval fixtures. It does
not claim broad benchmark coverage or a full dataset evaluation run.

`phase10_ops_surface/` demonstrates local surface reachability. It does not
claim production TLS, identity, or ingress hardening. Its Prometheus and Grafana
artifacts prove local scrape/dashboard wiring, not production observability
operations.

Generated artifacts should stay close to the behavior they support. If a new
artifact does not clarify current runtime behavior, it should not be added to the
stable saved bundle.
