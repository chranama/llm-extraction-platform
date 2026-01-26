#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-llm}"
API_SVC="${API_SVC:-api}"
LOCAL_PORT="${LOCAL_PORT:-8000}"
REMOTE_PORT="${REMOTE_PORT:-8000}"
API_BASE="http://localhost:${LOCAL_PORT}"
API_KEY="${API_KEY:-}"

AUTH_HEADER=()
if [[ -n "${API_KEY}" ]]; then
  AUTH_HEADER=(-H "X-API-Key: ${API_KEY}")
fi

need() { command -v "$1" >/dev/null 2>&1 || { echo "missing $1"; exit 1; }; }
need kubectl
need curl
need python

echo "Port-forward svc/${API_SVC} ${LOCAL_PORT}:${REMOTE_PORT} (ns=${NAMESPACE})"
kubectl -n "${NAMESPACE}" port-forward "svc/${API_SVC}" "${LOCAL_PORT}:${REMOTE_PORT}" >/tmp/k8s_pf.log 2>&1 &
pf_pid=$!
trap 'kill ${pf_pid} >/dev/null 2>&1 || true' EXIT

# Wait for port-forward to be live
for _ in $(seq 1 50); do
  if curl -fsS "${API_BASE}/healthz" >/dev/null 2>&1; then
    break
  fi
  sleep 0.2
done

echo "1) /healthz"
curl -fsS "${API_BASE}/healthz" >/dev/null
echo "OK"

echo "2) /v1/models"
models_json="$(curl -fsS "${API_BASE}/v1/models" "${AUTH_HEADER[@]}")"
echo "${models_json}" | python - <<'PY'
import json,sys
x=json.load(sys.stdin)

dep=x.get("deployment_capabilities") or {}
gen=dep.get("generate")
ext=dep.get("extract")

assert gen is True, f"deployment_capabilities.generate expected True, got {gen}"
assert ext is False, f"deployment_capabilities.extract expected False, got {ext}"

models=x.get("models") or []
assert models, "no models returned"

bad=[]
for m in models:
  caps=m.get("capabilities") or {}
  if caps.get("generate") is not True or caps.get("extract") is not False:
    bad.append((m.get("id"), caps))

assert not bad, f"models with non-generate-only caps: {bad}"
print("OK: generate-only verified via /v1/models")
PY

echo "3) POST /v1/generate (minimal)"
gen_payload='{"prompt":"ping","max_tokens":8}'
curl -fsS -X POST "${API_BASE}/v1/generate" \
  "${AUTH_HEADER[@]}" \
  -H "Content-Type: application/json" \
  --data "${gen_payload}" >/dev/null
echo "OK"

echo "4) POST /v1/extract must NOT succeed"
extract_payload='{"schema_id":"invoice_v1","text":"probe","cache":false,"repair":false}'
code="$(curl -sS -o /dev/null -w "%{http_code}" \
  -X POST "${API_BASE}/v1/extract" \
  "${AUTH_HEADER[@]}" \
  -H "Content-Type: application/json" \
  --data "${extract_payload}" || true
)"
if [[ "${code}" == "200" ]]; then
  echo "FAIL: /v1/extract returned 200 but should be disabled"
  exit 1
fi
echo "OK: /v1/extract disabled (HTTP ${code})"

echo "✅ k8s smoke passed"