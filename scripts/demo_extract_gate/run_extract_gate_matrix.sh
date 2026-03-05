#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../" && pwd)"
cd "$ROOT"

SCRIPT_NAME="$(basename "$0")"

usage() {
  cat <<'EOF'
Usage:
  run_extract_gate_matrix.sh [--skip-host] [--skip-docker]

Environment knobs:
  API_PORT            API port to probe (default: 8000)
  POSTGRES_HOST_PORT  host postgres published port (default: 5433)
  REDIS_HOST_PORT     host redis published port (default: 6380)
  HOST_PROFILE        models profile for host matrix (default: host-llama)
  DOCKER_PROFILE      models profile for docker matrix (default: docker-llama)
  DEMO_MODEL_ID       shared fallback demo model id
  HOST_DEMO_MODEL_ID  host matrix model id override
  DOCKER_DEMO_MODEL_ID docker matrix model id override
  RUN_TAG             output run tag under traffic_out/
  OUT_DIR             explicit output directory
EOF
}

if [[ -f ".env.docker" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".env.docker"
  set +a
fi

DEFAULT_LLAMA_DEMO_MODEL_ID="llama.cpp/SmolLM2-360M-Instruct-Q8_0-GGUF"
DEFAULT_TRANSFORMERS_DEMO_MODEL_ID="sshleifer/tiny-gpt2"
DEMO_MODEL_ID="${DEMO_MODEL_ID:-$DEFAULT_LLAMA_DEMO_MODEL_ID}"
API_PORT="${API_PORT:-8000}"
POSTGRES_HOST_PORT="${POSTGRES_HOST_PORT:-5433}"
REDIS_HOST_PORT="${REDIS_HOST_PORT:-6380}"
HOST_PROFILE="${HOST_PROFILE:-host-llama}"
DOCKER_PROFILE="${DOCKER_PROFILE:-docker-llama}"
HOST_DEMO_MODEL_ID="${HOST_DEMO_MODEL_ID:-$DEMO_MODEL_ID}"
DOCKER_DEMO_MODEL_ID="${DOCKER_DEMO_MODEL_ID:-$DEMO_MODEL_ID}"

if [[ "$HOST_PROFILE" == "host-transformers" && "$HOST_DEMO_MODEL_ID" == "$DEFAULT_LLAMA_DEMO_MODEL_ID" ]]; then
  HOST_DEMO_MODEL_ID="$DEFAULT_TRANSFORMERS_DEMO_MODEL_ID"
fi
if [[ "$DOCKER_PROFILE" == "docker-transformers" && "$DOCKER_DEMO_MODEL_ID" == "$DEFAULT_LLAMA_DEMO_MODEL_ID" ]]; then
  DOCKER_DEMO_MODEL_ID="$DEFAULT_TRANSFORMERS_DEMO_MODEL_ID"
fi

RUN_TAG="${RUN_TAG:-phase41_$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_DIR="${OUT_DIR:-$ROOT/traffic_out/$RUN_TAG}"
mkdir -p "$OUT_DIR"

SKIP_HOST=0
SKIP_DOCKER=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-host) SKIP_HOST=1; shift ;;
    --skip-docker) SKIP_DOCKER=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || { echo "missing required command: $1" >&2; exit 2; }
}

need_cmd uv
need_cmd docker
need_cmd curl

server_pid=""
CURRENT_PHASE="init"
DIAG_DIR="$OUT_DIR/diagnostics"
mkdir -p "$DIAG_DIR"

write_http_snapshot() {
  local label="$1"
  local path="$2"
  set +e
  curl -sS "http://127.0.0.1:${API_PORT}${path}" >"$DIAG_DIR/${label}.json" 2>"$DIAG_DIR/${label}.err"
  local ec=$?
  set -e
  if [[ "$ec" -ne 0 ]]; then
    echo "curl_failed:$ec" >"$DIAG_DIR/${label}.status"
  else
    echo "ok" >"$DIAG_DIR/${label}.status"
  fi
}

capture_runtime_snapshots() {
  local prefix="$1"
  write_http_snapshot "${prefix}_readyz" "/readyz"
  write_http_snapshot "${prefix}_modelz" "/modelz"
  write_http_snapshot "${prefix}_models" "/v1/models"
}

emit_log_grep() {
  local src="$1"
  local out="$2"
  if [[ -f "$src" ]]; then
    rg -n "models: loaded|requested_profile|used_profile|deployment_key|gated repo|model load failed|capability_disabled|policy:" "$src" >"$out" || true
  fi
}

dump_diagnostics() {
  local reason="$1"
  set +e
  {
    echo "reason=${reason}"
    echo "phase=${CURRENT_PHASE}"
    echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "script=${SCRIPT_NAME}"
    echo "api_port=${API_PORT}"
    echo "host_profile=${HOST_PROFILE}"
    echo "docker_profile=${DOCKER_PROFILE}"
    echo "host_demo_model_id=${HOST_DEMO_MODEL_ID}"
    echo "docker_demo_model_id=${DOCKER_DEMO_MODEL_ID}"
  } >"$DIAG_DIR/failure_context.env"

  capture_runtime_snapshots "failure_${CURRENT_PHASE}"

  compose_llmep ps >"$DIAG_DIR/compose_ps.txt" 2>&1 || true
  docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' >"$DIAG_DIR/docker_ps.txt" 2>&1 || true

  for cname in llmep-server-1 llmep-server_llama-1 llmep-server_llama_host-1; do
    docker logs "$cname" >"$DIAG_DIR/${cname}.log" 2>&1 || true
  done

  emit_log_grep "$OUT_DIR/host_pass_server.log" "$DIAG_DIR/host_pass_server.grep.log"
  emit_log_grep "$OUT_DIR/host_fail_server.log" "$DIAG_DIR/host_fail_server.grep.log"
  emit_log_grep "$DIAG_DIR/llmep-server_llama_host-1.log" "$DIAG_DIR/server_llama_host.grep.log"
  emit_log_grep "$DIAG_DIR/llmep-server_llama-1.log" "$DIAG_DIR/server_llama.grep.log"
  set -e
}

on_error() {
  local exit_code="$1"
  local line_no="$2"
  echo "ERROR: ${SCRIPT_NAME} failed at line ${line_no} during phase '${CURRENT_PHASE}' (exit=${exit_code})." >&2
  dump_diagnostics "exit=${exit_code} line=${line_no}"
  echo "Diagnostics written to: ${DIAG_DIR}" >&2
  exit "$exit_code"
}

trap 'on_error $? $LINENO' ERR

cleanup_server() {
  if [[ -n "${server_pid:-}" ]] && kill -0 "$server_pid" >/dev/null 2>&1; then
    kill "$server_pid" >/dev/null 2>&1 || true
    wait "$server_pid" >/dev/null 2>&1 || true
  fi
  server_pid=""
}
trap cleanup_server EXIT

compose_llmep() {
  docker compose -f deploy/compose/docker-compose.yml --env-file .env.docker -p llmep "$@"
}

wait_ready() {
  local tries="${1:-80}"
  for i in $(seq 1 "$tries"); do
    if curl -fsS "http://127.0.0.1:${API_PORT}/readyz" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

upsert_host_api_key() {
  if [[ -z "${API_KEY:-}" ]]; then
    echo "API_KEY is required for host checks" >&2
    exit 2
  fi
  docker exec -i llmep-postgres_host-1 psql -U llm -d llm -v ON_ERROR_STOP=1 \
    -c "INSERT INTO api_keys (key, active, quota_used, created_at) VALUES ('$API_KEY', true, 0, now()) ON CONFLICT (key) DO UPDATE SET active = EXCLUDED.active;" \
    >/dev/null
}

generate_artifacts_for_profile() {
  CURRENT_PHASE="artifacts_${5}"
  local profile="$1"
  local pass_out="$2"
  local fail_out="$3"
  local model_id="$4"
  local run_prefix="$5"
  uv run sim artifacts demo-eval --fixture pass --run-id "${run_prefix}_pass" --model-id "$model_id" >/dev/null
  uv run sim artifacts demo-eval --fixture fail --run-id "${run_prefix}_fail" --model-id "$model_id" >/dev/null
  uv run sim artifacts onboarding-demo --fixture pass --model-id "$model_id" --models-profile "$profile" --eval-run-dir "results/extract/${run_prefix}_pass" --out-models-yaml "$pass_out" >/dev/null
  uv run sim artifacts onboarding-demo --fixture fail --model-id "$model_id" --models-profile "$profile" --eval-run-dir "results/extract/${run_prefix}_fail" --out-models-yaml "$fail_out" >/dev/null
}

run_host_matrix() {
  CURRENT_PHASE="host_matrix"
  local profile="$1"
  local pass_models="$2"
  local fail_models="$3"
  local model_id="$4"

  cleanup_server
  compose_llmep down || true

  CURRENT_PHASE="host_infra_up"
  POSTGRES_HOST_PORT="$POSTGRES_HOST_PORT" \
  REDIS_HOST_PORT="$REDIS_HOST_PORT" \
  compose_llmep --profile infra-host up -d --remove-orphans --force-recreate
  upsert_host_api_key

  export DATABASE_URL="postgresql+asyncpg://llm:llm@127.0.0.1:${POSTGRES_HOST_PORT}/llm"
  export REDIS_ENABLED=1
  export REDIS_URL="redis://127.0.0.1:${REDIS_HOST_PORT}/0"
  export APP_PROFILE=host
  export MODELS_PROFILE="$profile"
  export POLICY_DECISION_PATH=""
  export LLAMA_SERVER_URL="http://127.0.0.1:8080"
  export SCHEMAS_DIR="$ROOT/schemas/model_output"

  # Host PASS
  CURRENT_PHASE="host_pass_start"
  MODELS_YAML="$ROOT/$pass_models" uv run --project server llm --host 0.0.0.0 --port "$API_PORT" >"$OUT_DIR/host_pass_server.log" 2>&1 &
  server_pid="$!"
  sleep 1
  if ! kill -0 "$server_pid" >/dev/null 2>&1; then
    echo "host PASS server exited before ready" >&2
    tail -n 80 "$OUT_DIR/host_pass_server.log" >&2 || true
    exit 1
  fi
  wait_ready 80 || { echo "host PASS server did not become ready" >&2; exit 1; }
  capture_runtime_snapshots "host_pass"
  CURRENT_PHASE="host_pass_checks"
  uv run sim --base-url "http://127.0.0.1:${API_PORT}" traffic runtime-proof --model-id "$model_id" --expect-policy-source none --expect-policy-enable-extract none --expect-model-extract true >"$OUT_DIR/host_pass_runtime.json"
  uv run sim --base-url "http://127.0.0.1:${API_PORT}" traffic extract-gate-check --model-id "$model_id" --expect allow --allow-model-errors --expect-model-extract true >"$OUT_DIR/host_pass_extract.json"

  cleanup_server

  # Host FAIL
  CURRENT_PHASE="host_fail_start"
  MODELS_YAML="$ROOT/$fail_models" uv run --project server llm --host 0.0.0.0 --port "$API_PORT" >"$OUT_DIR/host_fail_server.log" 2>&1 &
  server_pid="$!"
  sleep 1
  if ! kill -0 "$server_pid" >/dev/null 2>&1; then
    echo "host FAIL server exited before ready" >&2
    tail -n 80 "$OUT_DIR/host_fail_server.log" >&2 || true
    exit 1
  fi
  wait_ready 80 || { echo "host FAIL server did not become ready" >&2; exit 1; }
  capture_runtime_snapshots "host_fail"
  CURRENT_PHASE="host_fail_checks"
  uv run sim --base-url "http://127.0.0.1:${API_PORT}" traffic runtime-proof --model-id "$model_id" --expect-policy-source none --expect-policy-enable-extract none --expect-model-extract false >"$OUT_DIR/host_fail_runtime.json"
  uv run sim --base-url "http://127.0.0.1:${API_PORT}" traffic extract-gate-check --model-id "$model_id" --expect block --expect-model-extract false >"$OUT_DIR/host_fail_extract.json"
  cleanup_server
}

run_docker_matrix() {
  CURRENT_PHASE="docker_matrix"
  local profile="$1"
  local pass_models="$2"
  local fail_models="$3"
  local model_id="$4"

  cleanup_server
  compose_llmep down || true

  # Docker PASS
  CURRENT_PHASE="docker_pass_up"
  MODELS_PROFILE="$profile" \
  MODELS_YAML="/app/config/$(basename "$pass_models")" \
  POLICY_DECISION_PATH="" \
  LLAMA_SERVER_URL="http://host.docker.internal:8080" \
  DATABASE_URL="postgresql+asyncpg://llm:llm@postgres:5432/llm" \
  REDIS_ENABLED="1" \
  REDIS_URL="redis://redis:6379/0" \
  CONTAINER_MEMORY_BYTES="0" \
  SCHEMAS_DIR="/app/schemas/model_output" \
  API_KEY="${API_KEY:-}" \
  compose_llmep --profile infra --profile server-llama-host up -d --remove-orphans --force-recreate

  wait_ready 80 || { echo "docker PASS server did not become ready" >&2; exit 1; }
  capture_runtime_snapshots "docker_pass"
  CURRENT_PHASE="docker_pass_checks"
  uv run sim --base-url "http://127.0.0.1:${API_PORT}" traffic runtime-proof --model-id "$model_id" --expect-policy-source none --expect-policy-enable-extract none --expect-model-extract true >"$OUT_DIR/docker_pass_runtime.json"
  uv run sim --base-url "http://127.0.0.1:${API_PORT}" traffic extract-gate-check --model-id "$model_id" --expect allow --allow-model-errors --expect-model-extract true >"$OUT_DIR/docker_pass_extract.json"

  # Docker FAIL
  CURRENT_PHASE="docker_fail_up"
  MODELS_PROFILE="$profile" \
  MODELS_YAML="/app/config/$(basename "$fail_models")" \
  POLICY_DECISION_PATH="" \
  LLAMA_SERVER_URL="http://host.docker.internal:8080" \
  DATABASE_URL="postgresql+asyncpg://llm:llm@postgres:5432/llm" \
  REDIS_ENABLED="1" \
  REDIS_URL="redis://redis:6379/0" \
  CONTAINER_MEMORY_BYTES="0" \
  SCHEMAS_DIR="/app/schemas/model_output" \
  API_KEY="${API_KEY:-}" \
  compose_llmep --profile infra --profile server-llama-host up -d --remove-orphans --force-recreate

  wait_ready 80 || { echo "docker FAIL server did not become ready" >&2; exit 1; }
  capture_runtime_snapshots "docker_fail"
  CURRENT_PHASE="docker_fail_checks"
  uv run sim --base-url "http://127.0.0.1:${API_PORT}" traffic runtime-proof --model-id "$model_id" --expect-policy-source none --expect-policy-enable-extract none --expect-model-extract false >"$OUT_DIR/docker_fail_runtime.json"
  uv run sim --base-url "http://127.0.0.1:${API_PORT}" traffic extract-gate-check --model-id "$model_id" --expect block --expect-model-extract false >"$OUT_DIR/docker_fail_extract.json"
}

HOST_PROFILE_TAG="${HOST_PROFILE//-/_}"
DOCKER_PROFILE_TAG="${DOCKER_PROFILE//-/_}"
HOST_PASS_MODELS="config/models.patched.${HOST_PROFILE_TAG}.pass.yaml"
HOST_FAIL_MODELS="config/models.patched.${HOST_PROFILE_TAG}.fail.yaml"
DOCKER_PASS_MODELS="config/models.patched.${DOCKER_PROFILE_TAG}.pass.yaml"
DOCKER_FAIL_MODELS="config/models.patched.${DOCKER_PROFILE_TAG}.fail.yaml"

generate_artifacts_for_profile "$HOST_PROFILE" "$HOST_PASS_MODELS" "$HOST_FAIL_MODELS" "$HOST_DEMO_MODEL_ID" "demo_extract_${HOST_PROFILE_TAG}"
generate_artifacts_for_profile "$DOCKER_PROFILE" "$DOCKER_PASS_MODELS" "$DOCKER_FAIL_MODELS" "$DOCKER_DEMO_MODEL_ID" "demo_extract_${DOCKER_PROFILE_TAG}"

if [[ "$SKIP_HOST" -eq 0 ]]; then
  run_host_matrix "$HOST_PROFILE" "$HOST_PASS_MODELS" "$HOST_FAIL_MODELS" "$HOST_DEMO_MODEL_ID"
fi
if [[ "$SKIP_DOCKER" -eq 0 ]]; then
  run_docker_matrix "$DOCKER_PROFILE" "$DOCKER_PASS_MODELS" "$DOCKER_FAIL_MODELS" "$DOCKER_DEMO_MODEL_ID"
fi

CURRENT_PHASE="manifest"
python3 "$ROOT/scripts/demo_extract_gate/write_evidence_manifest.py" --run-dir "$OUT_DIR" || true

CURRENT_PHASE="done"
echo "Phase 4.1 complete. Evidence written to: $OUT_DIR"
