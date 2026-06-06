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
- `tools/joint/inference_gateway_stack.sh`: runs the LLMEP plus
  `inference-serving-gateway` joint workflow and writes joint gateway artifacts.

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
- `joint_gateway/latest/`: compatibility deterministic gateway proof.
- `joint_gateway/observability_latest/`: sync and async extract through the
  companion gateway, request and trace identity propagation, gateway/backend
  metrics, admin trace inspection, OTel traces, and process logs from the joint
  host run.
- `joint_gateway/edge_controls_latest/`: gateway-owned route policy,
  unsupported route, request-size rejection, backend auth pass-through, and
  metrics artifacts.
- `joint_gateway/llama_extract_latest/`: real model-backed sync and async
  extraction through the gateway using CPU-only containerized `llama.cpp`.
- `joint_gateway/containerized_latest/`: LLMEP API, LLMEP worker, gateway,
  Postgres, and Redis running together on one local Compose network.
- `joint_gateway/containerized_llama_latest/`: LLMEP API, LLMEP worker,
  gateway, Postgres, Redis, and CPU-only `llama.cpp` running together on one
  local Compose network.
- `joint_gateway/kind_smoke_latest/`: Kubernetes-shaped joint deployment proof
  copied from the gateway repository's local kind workflow.

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

`joint_gateway/latest/` and `joint_gateway/observability_latest/` demonstrate
local edge/backend integration with a deterministic model profile. They do not
claim real-model extraction quality, cloud networking, TLS, production identity,
or AWS deployment.

`joint_gateway/llama_extract_latest/` demonstrates real model-backed extraction
through the gateway with CPU-only `llama.cpp`. It does not claim accelerated
inference or production throughput.

`joint_gateway/containerized_latest/` demonstrates local Compose service
networking across LLMEP and the gateway. It does not claim Kubernetes or cloud
readiness.

`joint_gateway/containerized_llama_latest/` demonstrates the fully
containerized local joint stack with real model-backed extraction. It does not
claim accelerated inference, production throughput, cloud ingress, TLS, or high
availability.

`joint_gateway/kind_smoke_latest/` demonstrates Kubernetes-shaped local
deployability. It does not claim AWS ingress, TLS, high availability, or managed
cloud operation.

Generated artifacts should stay close to the behavior they support. If a new
artifact does not clarify current runtime behavior, it should not be added to the
stable saved bundle.
