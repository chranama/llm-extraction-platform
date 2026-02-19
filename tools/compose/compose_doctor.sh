#!/usr/bin/env bash
set -euo pipefail

red() { printf "\033[31m%s\033[0m\n" "$*"; }
grn() { printf "\033[32m%s\033[0m\n" "$*"; }
ylw() { printf "\033[33m%s\033[0m\n" "$*"; }
blu() { printf "\033[34m%s\033[0m\n" "$*"; }

API_PORT="${API_PORT:-8000}"
UI_PORT="${UI_PORT:-5173}"
PGADMIN_PORT="${PGADMIN_PORT:-5050}"
PROM_PORT="${PROM_PORT:-9090}"
GRAFANA_PORT="${GRAFANA_PORT:-3000}"
PROM_HOST_PORT="${PROM_HOST_PORT:-9091}"

API_BASE="http://localhost:${API_PORT}"
API_KEY="${API_KEY:-}"

# Optional: allow doctor to validate compose config
ENV_FILE="${ENV_FILE:-.env}"
COMPOSE_YML="${COMPOSE_YML:-deploy/compose/docker-compose.yml}"

AUTH_HEADER=()
if [[ -n "${API_KEY}" ]]; then
  AUTH_HEADER=(-H "X-API-Key: ${API_KEY}")
fi

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || { red "❌ Missing required command: $1"; exit 1; }
}

port_in_use() {
  local port="$1"
  if command -v lsof >/dev/null 2>&1; then
    lsof -iTCP -sTCP:LISTEN -n -P 2>/dev/null | grep -q ":${port} "
    return $?
  elif command -v ss >/dev/null 2>&1; then
    ss -ltn 2>/dev/null | awk '{print $4}' | grep -q ":${port}$"
    return $?
  elif command -v netstat >/dev/null 2>&1; then
    netstat -an 2>/dev/null | grep -q "\.${port} .*LISTEN"
    return $?
  else
    return 2
  fi
}

check_port_status() {
  local port="$1"
  local label="$2"
  if port_in_use "$port"; then
    grn "✅ Port ${port} (${label}) is listening"
  else
    ylw "⚠️  Port ${port} (${label}) is not listening (service may be down)"
  fi
}

curl_json() {
  local url="$1"
  shift
  curl -fsS "$url" "$@"
}

curl_status() {
  local url="$1"
  shift
  curl -sS -o /dev/null -w "%{http_code}" "$url" "$@"
}

hr() { echo "--------------------------------------------------------------------------------"; }

main() {
  need_cmd curl
  need_cmd grep
  need_cmd sed
  need_cmd python3

  blu "compose doctor"
  hr

  echo "API_BASE=${API_BASE}"
  echo "ENV_FILE=${ENV_FILE}"
  echo "COMPOSE_YML=${COMPOSE_YML}"
  if [[ -n "${API_KEY}" ]]; then
    echo "API_KEY=*** (set)"
  else
    ylw "⚠️  API_KEY is not set. /v1/* checks may fail with 401."
  fi

  hr
  blu "Compose config (optional)"
  if command -v docker >/dev/null 2>&1 && [[ -f "${COMPOSE_YML}" ]]; then
    # best effort: don't hard fail on missing env file
    if docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_YML}" config >/dev/null 2>&1; then
      grn "✅ docker compose config OK"
    else
      ylw "⚠️  docker compose config failed (check YAML/env interpolation)"
    fi
  else
    ylw "⚠️  Skipping compose config check (docker or compose file not found)"
  fi

  hr
  blu "Port status (best-effort)"
  check_port_status "${API_PORT}" "api"
  check_port_status "${UI_PORT}" "ui"
  check_port_status "${PGADMIN_PORT}" "pgadmin"
  check_port_status "${PROM_PORT}" "prometheus"
  check_port_status "${GRAFANA_PORT}" "grafana"
  check_port_status "${PROM_HOST_PORT}" "prometheus_host"

  hr
  blu "HTTP checks"

  code="$(curl_status "${API_BASE}/healthz" || true)"
  [[ "${code}" == "200" ]] && grn "✅ /healthz OK" || ylw "⚠️  /healthz returned HTTP ${code}"

  code="$(curl_status "${API_BASE}/readyz" || true)"
  if [[ "${code}" == "200" ]]; then
    grn "✅ /readyz OK"
  else
    if [[ -n "${API_KEY}" ]]; then
      code2="$(curl_status "${API_BASE}/readyz" "${AUTH_HEADER[@]}" || true)"
      [[ "${code2}" == "200" ]] && grn "✅ /readyz OK (auth)" || ylw "⚠️  /readyz HTTP ${code} (no-auth) / ${code2} (auth)"
    else
      ylw "⚠️  /readyz returned HTTP ${code}"
    fi
  fi

  echo
  blu "/modelz"
  out=""
  if out="$(curl_json "${API_BASE}/modelz" 2>/dev/null)"; then
    :
  elif [[ -n "${API_KEY}" ]] && out="$(curl_json "${API_BASE}/modelz" "${AUTH_HEADER[@]}" 2>/dev/null)"; then
    :
  else
    ylw "⚠️  Unable to fetch /modelz (may not exist, require auth, or API is down)."
    out=""
  fi

  if [[ -n "${out}" ]]; then
    # lightweight "is it JSON-ish?" check
    if [[ "${out}" =~ ^[[:space:]]*[\{\[] ]]; then
      echo "${out}" | sed 's/^/  /'
    else
      ylw "⚠️  /modelz returned non-JSON payload:"
      echo "${out}" | sed 's/^/  /'
    fi
  fi

  hr
  blu "Infer 'extract enabled' by probing /v1/schemas and /v1/extract"

  schemas_code="$(curl_status "${API_BASE}/v1/schemas" "${AUTH_HEADER[@]}" || true)"
  if [[ "${schemas_code}" != "200" ]]; then
    ylw "⚠️  GET /v1/schemas returned HTTP ${schemas_code} (expected 200)."
    ylw "   If this is 401: set API_KEY."
    hr
    grn "✅ compose doctor finished (partial)."
    return 0
  fi

  schemas_json="$(curl_json "${API_BASE}/v1/schemas" "${AUTH_HEADER[@]}")"
  schema_count="$(echo "${schemas_json}" | python3 - <<'PY'
import json,sys
try:
  x=json.load(sys.stdin)
  print(len(x) if isinstance(x,list) else 0)
except Exception:
  print(0)
PY
)"
  echo "Schemas returned: ${schema_count}"

  if [[ "${schema_count}" -le 0 ]]; then
    ylw "⚠️  No schemas returned; extraction likely not usable (or schemas not configured)."
    hr
    grn "✅ compose doctor finished."
    return 0
  fi

  # Prefer invoice_v1 if present, else first item
  schema_id="$(echo "${schemas_json}" | python3 - <<'PY'
import json,sys
x=json.load(sys.stdin)
sid=""
if isinstance(x,list) and x:
  ids=[]
  for d in x:
    if isinstance(d,dict) and d.get("schema_id"):
      ids.append(str(d["schema_id"]))
  if "invoice_v1" in ids:
    sid="invoice_v1"
  elif ids:
    sid=ids[0]
print(sid)
PY
)"
  if [[ -z "${schema_id}" ]]; then
    ylw "⚠️  Could not determine schema_id from /v1/schemas payload."
    hr
    grn "✅ compose doctor finished."
    return 0
  fi

  tmp_out="$(mktemp -t compose_doctor_extract.XXXXXX.json)"
  trap 'rm -f "${tmp_out}"' EXIT

  extract_payload="$(cat <<EOF
{"schema_id":"${schema_id}","text":"probe","cache":false,"repair":false}
EOF
)"
  extract_code="$(curl -sS -o "${tmp_out}" -w "%{http_code}" \
    -X POST "${API_BASE}/v1/extract" \
    "${AUTH_HEADER[@]}" \
    -H "Content-Type: application/json" \
    --data "${extract_payload}" || true
  )"

  if [[ "${extract_code}" == "200" ]]; then
    grn "✅ POST /v1/extract OK (schema_id=${schema_id})"
  else
    ylw "⚠️  POST /v1/extract returned HTTP ${extract_code} (schema_id=${schema_id})"
    ylw "   Likely generate-only MODELS_YAML or routing rejected it."
    ylw "   Body (first 300 chars): $(head -c 300 "${tmp_out}" | tr '\n' ' ')"
  fi

  hr
  grn "✅ compose doctor finished."
}

main "$@"