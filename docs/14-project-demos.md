# 14) Project Demos

This document tracks reproducible project demos.

Current scope:
- Demo 1: Generate clamp (implemented below)
- Demo 2: Extract gate (to be added next)

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

Pending. This section will be added next.
