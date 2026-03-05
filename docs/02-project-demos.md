# 02) Project Demos

This document tracks reproducible project demos.

Current scope:
- Demo 1: Generate clamp (implemented below)
- Demo 2: Extract gate (implemented below)

## Demo 1: Generate Clamp

### Goal
Show both states clearly:
- Control: `demo-baseline` does **not** produce a clamp decision.
- Clamp: `demo-clamp` **does** produce a clamp decision that is enforced on `/v1/generate`.

### What changed
The policy CLI now uses:
- `policy runtime-decision` (new)

instead of the previous:
- `policy run` (old)

### Prerequisites
- Docker running
- `uv` available
- `.env.docker` present with `API_KEY`

### Step 1: Start demo stack
```bash
uv run llmctl --project-name llmep \
  compose --defaults-profile docker+llama+models-llama \
  --env-override-file .env.docker \
  up --profiles infra llama server-llama ui -d --build --remove-orphans
```

Optional status check:
```bash
uv run llmctl --project-name llmep \
  compose --defaults-profile docker+llama+models-llama \
  --env-override-file .env.docker \
  ps --profiles infra llama server-llama ui
```

### Step 2: Clear previous artifacts
```bash
rm -f slo_out/generate/latest.json policy_out/latest.json
```

### Step 3: Phase A (control) run baseline traffic
```bash
set -a; source .env.docker; set +a
uv run sim traffic demo-baseline \
  --seconds 10 \
  --model-id "llama.cpp/SmolLM2-360M-Instruct-Q8_0-GGUF"
```

Expected:
- `summary.ok: true`
- p95 should stay below the portable clamp threshold (`600ms`)

### Step 4: Export generate SLO snapshot
```bash
set -a; source .env.docker; set +a
curl -s -X POST \
  "http://127.0.0.1:8000/v1/admin/slo/generate/write?window_seconds=60&out_path=/tmp/runtime_generate_slo.json" \
  -H "X-API-Key: $API_KEY" \
  | tee /tmp/slo_write_resp.json >/dev/null

jq '.payload' /tmp/slo_write_resp.json > slo_out/generate/latest.json
```

Quick check:
```bash
python - <<'PY'
import json
p=json.load(open("slo_out/generate/latest.json"))
print("generated_at:", p["generated_at"])
print("p95:", p["totals"]["latency_ms"]["p95"])
print("error_rate:", p["totals"]["errors"]["rate"])
print("total:", p["totals"]["requests"]["total"])
PY
```

### Step 5: Compute policy decision for baseline (new CLI)
```bash
uv run --project policy policy runtime-decision \
  --pipeline generate_clamp_only \
  --thresholds-root policy/src/llm_policy/thresholds \
  --generate-threshold-profile generate/portable \
  --generate-slo-path slo_out/generate/latest.json \
  --artifact-out policy_out/latest.json \
  --report text
```

Expected in output:
- `status=allow`
- reason includes `generate_slo_no_clamp`

### Step 6: Verify baseline artifact (no clamp)
```bash
python - <<'PY'
import json
p=json.load(open("policy_out/latest.json"))
print("schema_version:", p.get("schema_version"))
print("pipeline:", p.get("pipeline"))
print("status:", p.get("status"), "ok:", p.get("ok"))
print("generate_max_new_tokens_cap:", p.get("generate_max_new_tokens_cap"))
print("generate_thresholds_profile:", p.get("generate_thresholds_profile"))
print("metrics.generate_slo_latency_p95_ms:", p.get("metrics", {}).get("generate_slo_latency_p95_ms"))
PY
```

Expected:
- `generate_max_new_tokens_cap: None`

### Step 7: Phase B (clamp) run clamp scenario
```bash
set -a; source .env.docker; set +a
uv run sim traffic demo-clamp \
  --seconds 15 --rps 2 --concurrency 4 \
  --prompt-size long --no-cache \
  --model-id "llama.cpp/SmolLM2-360M-Instruct-Q8_0-GGUF"
```

Expected:
- `summary.ok: true`
- p95 well above threshold
- policy artifact rewritten with non-null `generate_max_new_tokens_cap` (typically `128`)

### Step 8: Reload policy into server
```bash
set -a; source .env.docker; set +a
uv run sim traffic admin policy-reload | jq .
```

Expected:
- `generate_max_new_tokens_cap` is set (for example `128`)

### Step 9: Prove clamp on `/v1/generate`
```bash
set -a; source .env.docker; set +a
curl -s http://127.0.0.1:8000/v1/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{
    "model_id": "llama.cpp/SmolLM2-360M-Instruct-Q8_0-GGUF",
    "prompt": "Write a long story about a robot learning to dance.",
    "max_new_tokens": 512
  }' | jq .
```

Acceptance signal:
- `"clamped": true`
- `"requested_max_new_tokens": 512`
- `"effective_max_new_tokens"` equals policy cap (for example `128`)

### Troubleshooting

#### Baseline unexpectedly clamps

Symptoms:
- `demo-baseline` policy decision has non-null `generate_max_new_tokens_cap`
- reasons include `generate_slo_latency_high` or high error rate conditions

Checks:
```bash
python - <<'PY'
import json
p=json.load(open("slo_out/generate/latest.json"))
print("p95:", p["totals"]["latency_ms"]["p95"])
print("error_rate:", p["totals"]["errors"]["rate"])
print("total:", p["totals"]["requests"]["total"])
PY
```

```bash
set -a; source .env.docker; set +a
curl -s http://127.0.0.1:8000/readyz | jq '.deployment.routing.backend_info.config'
```

Expected:
- p95 below `600ms` for baseline window
- error_rate near `0.0`
- `server_url` points to compose llama (`http://llama_server:8080`)

Fixes:
- restart `server_llama` to clear stale in-memory history, then rerun baseline
- use a narrow window for SLO export (`window_seconds=30` or `60`)
- ensure baseline uses control defaults (do not increase `--rps` / `--concurrency`)

#### Clamp does not trigger

Symptoms:
- after `demo-clamp`, policy decision still has `generate_max_new_tokens_cap: None`

Checks:
```bash
python - <<'PY'
import json
p=json.load(open("policy_out/latest.json"))
print("cap:", p.get("generate_max_new_tokens_cap"))
print("reasons:", [r.get("code") for r in p.get("reasons", [])])
print("p95_metric:", p.get("metrics", {}).get("generate_slo_latency_p95_ms"))
PY
```

```bash
uv run --project policy policy runtime-decision --help
```

Expected:
- using `policy runtime-decision` (new CLI)
- clamp reason `generate_slo_latency_high` when p95 crosses profile threshold

Fixes:
- rerun clamp phase with stronger load:
  - `--seconds 15 --rps 2 --concurrency 4 --prompt-size long --no-cache`
- verify profile is `--generate-threshold-profile generate/portable`
- rerun `uv run sim traffic admin policy-reload` before clamp verification request

## Demo 2: Extract Gate

### Goal
Show that offline onboarding decisions control extract availability through patched `models.yaml` artifacts.

Acceptance signal:
- PASS artifact => extract capability enabled => `/v1/extract` not capability-blocked.
- FAIL artifact => extract capability disabled => `/v1/extract` capability-blocked.

Unlike Demo 1 (runtime policy decision), this demo is artifact-driven:
- onboarding writes model capabilities into `config/models.patched.*.yaml`
- server enforces those capabilities at request time.

### Deployment Scope
Validated deployment shapes:
1. host + transformers (`host-transformers`)
2. host + llama (`host-llama`)
3. docker + host llama-server (`docker-llama` via `server-llama-host`)

### Prerequisites
- `uv` available
- Docker running
- `.env.docker` present with `API_KEY`
- host llama-server reachable at `http://127.0.0.1:8080` for llama-backed runs

Schema/contract split used by this demo:
- schema specs: `schemas/`
- type/validation contracts: `contracts/` (`llm_contracts`)

### Recommended Quickstart
Use the operational runners from `scripts/demo_extract_gate/`.

Host transformers only:
```bash
scripts/demo_extract_gate/run_host_transformers.sh
```

Docker llama only:
```bash
scripts/demo_extract_gate/run_docker_llama.sh
```

Full matrix (host + docker):
```bash
scripts/demo_extract_gate/run_extract_gate_matrix.sh
```

Success criteria for each run:
- `<out_dir>/*_runtime.json` has `"ok": true`
- `<out_dir>/*_extract.json` has `"ok": true`
- PASS runtime shows `model_extract_capability: true`
- FAIL runtime shows `model_extract_capability: false`

### Manual Deep-Dive (Optional)
Use this if you want full control of each step.

Set shared env:
```bash
set -a; source .env.docker; set +a
export HOST_DEMO_MODEL_ID="sshleifer/tiny-gpt2"
export DOCKER_DEMO_MODEL_ID="llama.cpp/SmolLM2-360M-Instruct-Q8_0-GGUF"
export API_PORT=8000
export POLICY_DECISION_PATH=""
```

Build deterministic eval fixtures:
```bash
uv run sim artifacts demo-eval --fixture pass --run-id demo_extract_host_transformers_pass --model-id "$HOST_DEMO_MODEL_ID"
uv run sim artifacts demo-eval --fixture fail --run-id demo_extract_host_transformers_fail --model-id "$HOST_DEMO_MODEL_ID"
uv run sim artifacts demo-eval --fixture pass --run-id demo_extract_docker_llama_pass --model-id "$DOCKER_DEMO_MODEL_ID"
uv run sim artifacts demo-eval --fixture fail --run-id demo_extract_docker_llama_fail --model-id "$DOCKER_DEMO_MODEL_ID"
```

Generate patched model artifacts:
```bash
uv run sim artifacts onboarding-demo --fixture pass --model-id "$HOST_DEMO_MODEL_ID" --models-profile host-transformers --eval-run-dir results/extract/demo_extract_host_transformers_pass --out-models-yaml config/models.patched.host_transformers.pass.yaml
uv run sim artifacts onboarding-demo --fixture fail --model-id "$HOST_DEMO_MODEL_ID" --models-profile host-transformers --eval-run-dir results/extract/demo_extract_host_transformers_fail --out-models-yaml config/models.patched.host_transformers.fail.yaml
uv run sim artifacts onboarding-demo --fixture pass --model-id "$DOCKER_DEMO_MODEL_ID" --models-profile docker-llama --eval-run-dir results/extract/demo_extract_docker_llama_pass --out-models-yaml config/models.patched.docker_llama.pass.yaml
uv run sim artifacts onboarding-demo --fixture fail --model-id "$DOCKER_DEMO_MODEL_ID" --models-profile docker-llama --eval-run-dir results/extract/demo_extract_docker_llama_fail --out-models-yaml config/models.patched.docker_llama.fail.yaml
```

Verify host PASS artifact explicitly:
```bash
uv run sim artifacts verify-models-cap --path config/models.patched.host_transformers.pass.yaml --model-id "$HOST_DEMO_MODEL_ID" --models-profile host-transformers --expect-extract --expect-assessed
```

Verify host FAIL artifact explicitly:
```bash
uv run sim artifacts verify-models-cap --path config/models.patched.host_transformers.fail.yaml --model-id "$HOST_DEMO_MODEL_ID" --models-profile host-transformers --expect-no-extract --expect-assessed
```

Host PASS run:
```bash
uv run llmctl --project-name llmep compose --defaults-profile docker+llama+models-llama --env-override-file .env.docker up --profiles infra-host -d --remove-orphans --force-recreate
cd server
export DATABASE_URL="postgresql+asyncpg://llm:llm@127.0.0.1:5433/llm"
export REDIS_ENABLED=1
export REDIS_URL="redis://127.0.0.1:6380/0"
export MODELS_PROFILE=host-transformers
export MODELS_YAML="$(pwd)/../config/models.patched.host_transformers.pass.yaml"
export POLICY_DECISION_PATH=""
uv run python -m llm_server.cli serve --host 0.0.0.0 --port "$API_PORT"
```

In another terminal:
```bash
set -a; source .env.docker; set +a
uv run sim --base-url "http://127.0.0.1:${API_PORT}" traffic runtime-proof --model-id "$HOST_DEMO_MODEL_ID" --artifact-models-yaml config/models.patched.host_transformers.pass.yaml --artifact-models-profile host-transformers --expect-policy-source none --expect-policy-enable-extract none --expect-model-extract true
uv run sim --base-url "http://127.0.0.1:${API_PORT}" traffic extract-gate-check --model-id "$HOST_DEMO_MODEL_ID" --expect allow --allow-model-errors --expect-model-extract true
```

For FAIL, switch only the artifact path:
```bash
export MODELS_YAML="$(pwd)/../config/models.patched.host_transformers.fail.yaml"
```

Then rerun:
```bash
uv run sim --base-url "http://127.0.0.1:${API_PORT}" traffic runtime-proof --model-id "$HOST_DEMO_MODEL_ID" --artifact-models-yaml config/models.patched.host_transformers.fail.yaml --artifact-models-profile host-transformers --expect-policy-source none --expect-policy-enable-extract none --expect-model-extract false
uv run sim --base-url "http://127.0.0.1:${API_PORT}" traffic extract-gate-check --model-id "$HOST_DEMO_MODEL_ID" --expect block --expect-model-extract false
```

Docker PASS/FAIL run follows the same pattern with:
- `MODELS_PROFILE=docker-llama`
- `MODELS_YAML=/app/config/models.patched.docker_llama.(pass|fail).yaml`
- `LLAMA_SERVER_URL=http://host.docker.internal:8080`
- compose profile: `server-llama-host`

### Failure Diagnostics
Runner scripts automatically capture diagnostics in:
- `traffic_out/<run_tag>/diagnostics/`

Collected on each stage:
- `/readyz`, `/modelz`, `/v1/models` snapshots
- compose/docker process snapshots
- service logs + focused grep output for profile/model/deployment-key clues

### Troubleshooting
#### Artifact did not flip extract capability
- Re-run `sim artifacts verify-models-cap` on PASS/FAIL artifacts.
- Confirm `--eval-run-dir` and `--models-profile` match your target deployment profile.

#### Runtime is not using intended artifact
- Confirm `MODELS_YAML` path and `MODELS_PROFILE` are aligned.
- Run `sim traffic runtime-proof` with `--artifact-models-yaml` and `--artifact-models-profile`.
- Use recreate, not restart, when switching docker env/artifact wiring.

#### Offline demo contaminated by runtime policy
- Ensure `POLICY_DECISION_PATH=""` before launching server/compose.
- In runtime proof, assert:
  - `--expect-policy-source none`
  - `--expect-policy-enable-extract none`

#### Docker cannot reach host llama-server
- Verify host endpoint: `curl -s http://127.0.0.1:8080/health`
- Use `LLAMA_SERVER_URL=http://host.docker.internal:8080`
- On Linux, keep `extra_hosts: host.docker.internal:host-gateway`

## Evidence Manifest Contract (Both Demos)

Each demo should emit `traffic_out/<run>/evidence_manifest.json` with:
- demo id and run id
- control input summary
- expected behavior summary
- observed behavior summary
- pass/fail verdict
- key evidence file paths

This is used as the recruiter/hiring-manager quick proof artifact.

### Current Phase-3 Evidence Runs

- Extract gate concrete manifest:
  - `traffic_out/phase41_20260304T230327Z/evidence_manifest.json`
- Generate clamp control/clamp/adversarial manifests:
  - `traffic_out/phase3_generate_20260304/evidence_manifest_control.json`
  - `traffic_out/phase3_generate_20260304/evidence_manifest_clamp.json`
  - `traffic_out/phase3_generate_20260304/evidence_manifest_adversarial_mismatch.json`

## Adversarial / Failure-Case Proof (Required)

### Generate Clamp
Failure case to demonstrate:
- high-latency traffic should trigger clamp;
- if clamp does not trigger, manifest should record mismatch and include policy artifact path.

### Extract Gate
Failure case to demonstrate:
- FAIL onboarding artifact should capability-block extract;
- if extract is allowed under FAIL artifact, manifest should record mismatch and include runtime model capability snapshot.
