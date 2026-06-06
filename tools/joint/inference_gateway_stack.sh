#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLMEP_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CAREER_ROOT="$(cd "${LLMEP_ROOT}/.." && pwd)"

: "${ISG_REPO_ROOT:=${CAREER_ROOT}/inference-serving-gateway}"

BACKEND_SERVER_DIR="${LLMEP_ROOT}/server"
COMPOSE_FILE="${LLMEP_ROOT}/deploy/compose/docker-compose.yml"
RUNTIME_DIR="${LLMEP_ROOT}/.tmp/joint_gateway"
LOG_DIR="${RUNTIME_DIR}/logs"
PID_DIR="${RUNTIME_DIR}/pids"
STATE_DIR="${RUNTIME_DIR}/state"
ENV_FILE="${RUNTIME_DIR}/joint-gateway.env"
GATEWAY_BIN="${STATE_DIR}/inference-serving-gateway"

BACKEND_LOG="${LOG_DIR}/backend.log"
WORKER_LOG="${LOG_DIR}/worker.log"
GATEWAY_LOG="${LOG_DIR}/gateway.log"

BACKEND_PID_FILE="${PID_DIR}/backend.pid"
WORKER_PID_FILE="${PID_DIR}/worker.pid"
GATEWAY_PID_FILE="${PID_DIR}/gateway.pid"

: "${JOINT_APP_ROOT:=${LLMEP_ROOT}}"
: "${JOINT_APP_PROFILE:=test}"
: "${JOINT_MODELS_PROFILE:=gateway-proof}"
: "${JOINT_MODELS_YAML:=${LLMEP_ROOT}/proof/fixtures/models.gateway-proof.yaml}"
: "${JOINT_SCHEMAS_DIR:=${LLMEP_ROOT}/schemas/model_output}"

: "${JOINT_BACKEND_HOST:=127.0.0.1}"
: "${JOINT_BACKEND_PORT:=18090}"
: "${JOINT_GATEWAY_HOST:=127.0.0.1}"
: "${JOINT_GATEWAY_PORT:=18091}"
: "${JOINT_PG_HOST_PORT:=15433}"
: "${JOINT_REDIS_HOST_PORT:=16379}"
: "${JOINT_PROM_HOST_PORT:=19091}"
: "${JOINT_GRAFANA_PORT:=13000}"
: "${JOINT_OTEL_COLLECTOR_PORT:=14318}"
: "${JOINT_OTEL_COLLECTOR_HEALTH_PORT:=13135}"
: "${JOINT_JAEGER_PORT:=26688}"

: "${JOINT_DATABASE_URL:=postgresql+asyncpg://llm:llm@127.0.0.1:${JOINT_PG_HOST_PORT}/llm}"
: "${JOINT_REDIS_URL:=redis://127.0.0.1:${JOINT_REDIS_HOST_PORT}/0}"
: "${JOINT_PROOF_USER_KEY:=joint-proof-user-key}"
: "${JOINT_PROOF_ADMIN_KEY:=joint-proof-admin-key}"
: "${JOINT_WITH_OBS:=1}"
: "${JOINT_WITH_OTEL:=1}"
: "${JOINT_RUN_TESTS:=0}"
: "${JOINT_COMPOSE_PROJECT_NAME:=llmep_joint_gateway}"
: "${JOINT_ARTIFACT_DIR:=${LLMEP_ROOT}/proof/artifacts/joint_gateway/latest}"
: "${JOINT_OTEL_EXPORTER_OTLP_ENDPOINT:=http://127.0.0.1:${JOINT_OTEL_COLLECTOR_PORT}/v1/traces}"
: "${JOINT_BACKEND_OTEL_SERVICE_NAME:=llm-extraction-platform}"
: "${JOINT_WORKER_OTEL_SERVICE_NAME:=llm-extraction-platform-worker}"
: "${JOINT_GATEWAY_OTEL_SERVICE_NAME:=inference-serving-gateway}"

usage() {
  cat <<'EOF'
Usage:
  tools/joint/inference_gateway_stack.sh preflight
  tools/joint/inference_gateway_stack.sh up
  tools/joint/inference_gateway_stack.sh status
  tools/joint/inference_gateway_stack.sh proof
  tools/joint/inference_gateway_stack.sh down
  tools/joint/inference_gateway_stack.sh restart
  tools/joint/inference_gateway_stack.sh verify

Supported shape:
  - LLMEP API server on the host
  - LLMEP async extract worker on the host
  - LLMEP Postgres/Redis through the infra-host Compose profile
  - optional Prometheus/Grafana through the obs-host Compose profile
  - optional OTel Collector/Jaeger through the otel-host Compose profile
  - inference-serving-gateway built from the sibling repository

Common overrides:
  ISG_REPO_ROOT=/path/to/inference-serving-gateway
  JOINT_ARTIFACT_DIR=proof/artifacts/joint_gateway/latest
  JOINT_WITH_OBS=0
  JOINT_WITH_OTEL=0
  JOINT_RUN_TESTS=1
  JOINT_COMPOSE_PROJECT_NAME=llmep_joint_gateway
EOF
}

need_cmd() {
  local cmd="$1"
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "Missing required command: ${cmd}" >&2
    exit 1
  fi
}

need_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "Missing required file: ${path}" >&2
    exit 1
  fi
}

need_dir() {
  local path="$1"
  if [[ ! -d "${path}" ]]; then
    echo "Missing required directory: ${path}" >&2
    exit 1
  fi
}

ensure_docker_ready() {
  if ! docker info >/dev/null 2>&1; then
    echo "Docker is required for the joint gateway stack, but the daemon is not reachable." >&2
    exit 1
  fi
}

resolve_loopback_host() {
  local host="$1"
  case "${host}" in
    0.0.0.0|"::")
      printf '127.0.0.1\n'
      ;;
    *)
      printf '%s\n' "${host}"
      ;;
  esac
}

backend_url() {
  local host
  host="$(resolve_loopback_host "${JOINT_BACKEND_HOST}")"
  printf 'http://%s:%s\n' "${host}" "${JOINT_BACKEND_PORT}"
}

gateway_url() {
  local host
  host="$(resolve_loopback_host "${JOINT_GATEWAY_HOST}")"
  printf 'http://%s:%s\n' "${host}" "${JOINT_GATEWAY_PORT}"
}

server_python() {
  uv run --project "${BACKEND_SERVER_DIR}" python "$@"
}

ensure_layout() {
  mkdir -p "${LOG_DIR}" "${PID_DIR}" "${STATE_DIR}"
}

write_env_file() {
  ensure_layout
  cat >"${ENV_FILE}" <<EOF
ISG_REPO_ROOT=${ISG_REPO_ROOT}
APP_ROOT=${JOINT_APP_ROOT}
APP_PROFILE=${JOINT_APP_PROFILE}
MODELS_PROFILE=${JOINT_MODELS_PROFILE}
MODELS_YAML=${JOINT_MODELS_YAML}
SCHEMAS_DIR=${JOINT_SCHEMAS_DIR}
DATABASE_URL=${JOINT_DATABASE_URL}
REDIS_ENABLED=1
REDIS_URL=${JOINT_REDIS_URL}
EDGE_MODE=behind_gateway
BACKEND_URL=$(backend_url)
GATEWAY_URL=$(gateway_url)
OTEL_ENABLED=${JOINT_WITH_OTEL}
OTEL_EXPORTER_OTLP_ENDPOINT=${JOINT_OTEL_EXPORTER_OTLP_ENDPOINT}
ARTIFACT_DIR=${JOINT_ARTIFACT_DIR}
EOF
}

check_port_available() {
  local host="$1"
  local port="$2"
  if python3 - <<PY
import socket

host = "${host}"
port = int("${port}")
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(1.0)
try:
    sock.bind((host, port))
finally:
    sock.close()
PY
  then
    return 0
  fi
  echo "Port is not available: ${host}:${port}" >&2
  return 1
}

wait_for_tcp() {
  local host="$1"
  local port="$2"
  local attempts="${3:-80}"
  for _ in $(seq 1 "${attempts}"); do
    if python3 - <<PY >/dev/null 2>&1
import socket

sock = socket.socket()
sock.settimeout(1.0)
try:
    sock.connect(("${host}", int("${port}")))
finally:
    sock.close()
PY
    then
      return 0
    fi
    sleep 0.25
  done
  echo "Timed out waiting for TCP ${host}:${port}" >&2
  return 1
}

wait_for_url() {
  local url="$1"
  local attempts="${2:-80}"
  for _ in $(seq 1 "${attempts}"); do
    if curl -fsS "${url}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.25
  done
  echo "Timed out waiting for ${url}" >&2
  return 1
}

wait_for_database_url() {
  local database_url="$1"
  local attempts="${2:-80}"
  for _ in $(seq 1 "${attempts}"); do
    if (
      export DATABASE_URL="${database_url}"
      server_python - <<'PY'
import asyncio
import os

import asyncpg


def normalize_database_url(raw: str) -> str:
    if raw.startswith("postgresql+asyncpg://"):
        return "postgresql://" + raw[len("postgresql+asyncpg://"):]
    return raw


async def main() -> None:
    conn = await asyncpg.connect(normalize_database_url(os.environ["DATABASE_URL"]))
    try:
        await conn.execute("SELECT 1")
    finally:
        await conn.close()


asyncio.run(main())
PY
    ) >/dev/null 2>&1
    then
      return 0
    fi
    sleep 0.5
  done
  echo "Timed out waiting for database connection: ${database_url}" >&2
  return 1
}

compose_profiles() {
  local -a profiles=(infra-host)
  if [[ "${JOINT_WITH_OBS}" == "1" ]]; then
    profiles+=(obs-host)
  fi
  if [[ "${JOINT_WITH_OTEL}" == "1" ]]; then
    profiles+=(otel-host)
  fi
  printf '%s\n' "${profiles[@]}"
}

compose_with_profiles() {
  local -a cmd=(docker compose -f "${COMPOSE_FILE}")
  local profile
  while IFS= read -r profile; do
    cmd+=(--profile "${profile}")
  done < <(compose_profiles)
  cmd+=("$@")

  env \
    POSTGRES_HOST_PORT="${JOINT_PG_HOST_PORT}" \
    REDIS_HOST_PORT="${JOINT_REDIS_HOST_PORT}" \
    PROM_HOST_PORT="${JOINT_PROM_HOST_PORT}" \
    GRAFANA_PORT="${JOINT_GRAFANA_PORT}" \
    OTEL_COLLECTOR_HOST_PORT="${JOINT_OTEL_COLLECTOR_PORT}" \
    OTEL_COLLECTOR_HEALTH_HOST_PORT="${JOINT_OTEL_COLLECTOR_HEALTH_PORT}" \
    JAEGER_PORT="${JOINT_JAEGER_PORT}" \
    API_KEY="${JOINT_PROOF_USER_KEY}" \
    COMPOSE_PROJECT_NAME="${JOINT_COMPOSE_PROJECT_NAME}" \
    "${cmd[@]}"
}

is_pid_running() {
  local pid_file="$1"
  if [[ ! -f "${pid_file}" ]]; then
    return 1
  fi
  local pid
  pid="$(cat "${pid_file}")"
  [[ -n "${pid}" ]] || return 1
  kill -0 "${pid}" >/dev/null 2>&1
}

stop_pid_file() {
  local pid_file="$1"
  if ! is_pid_running "${pid_file}"; then
    rm -f "${pid_file}"
    return 0
  fi

  local pid
  pid="$(cat "${pid_file}")"
  kill "${pid}" >/dev/null 2>&1 || true
  for _ in $(seq 1 40); do
    if ! kill -0 "${pid}" >/dev/null 2>&1; then
      break
    fi
    sleep 0.25
  done
  if kill -0 "${pid}" >/dev/null 2>&1; then
    kill -9 "${pid}" >/dev/null 2>&1 || true
  fi
  rm -f "${pid_file}"
}

run_migrations() {
  (
    cd "${BACKEND_SERVER_DIR}"
    env \
      APP_ROOT="${JOINT_APP_ROOT}" \
      APP_PROFILE="${JOINT_APP_PROFILE}" \
      DATABASE_URL="${JOINT_DATABASE_URL}" \
      uv run python -m alembic -c "${BACKEND_SERVER_DIR}/alembic.ini" upgrade head
  )
}

seed_proof_keys() {
  (
    cd "${BACKEND_SERVER_DIR}"
    env \
      APP_ROOT="${JOINT_APP_ROOT}" \
      APP_PROFILE="${JOINT_APP_PROFILE}" \
      DATABASE_URL="${JOINT_DATABASE_URL}" \
      PROOF_USER_KEY="${JOINT_PROOF_USER_KEY}" \
      PROOF_ADMIN_KEY="${JOINT_PROOF_ADMIN_KEY}" \
      uv run python - <<'PY'
import asyncio
import os

from sqlalchemy import select

from llm_server.db.models import ApiKey, RoleTable
from llm_server.db.session import get_sessionmaker

PROOF_USER_KEY = os.environ["PROOF_USER_KEY"]
PROOF_ADMIN_KEY = os.environ["PROOF_ADMIN_KEY"]


async def ensure_role(session, name: str) -> RoleTable:
    role = (
        await session.execute(select(RoleTable).where(RoleTable.name == name))
    ).scalar_one_or_none()
    if role is None:
        role = RoleTable(name=name)
        session.add(role)
        await session.flush()
    return role


async def ensure_key(session, *, key: str, role_id: int | None) -> None:
    row = (
        await session.execute(select(ApiKey).where(ApiKey.key == key))
    ).scalar_one_or_none()
    if row is None:
        session.add(
            ApiKey(
                key=key,
                active=True,
                role_id=role_id,
                quota_monthly=None,
                quota_used=0,
            )
        )
        return
    if not row.active:
        row.active = True
    if row.role_id != role_id:
        row.role_id = role_id
    session.add(row)


async def main() -> None:
    sessionmaker = get_sessionmaker()
    async with sessionmaker() as session:
        admin = await ensure_role(session, "admin")
        standard = await ensure_role(session, "standard")
        await ensure_key(session, key=PROOF_USER_KEY, role_id=standard.id)
        await ensure_key(session, key=PROOF_ADMIN_KEY, role_id=admin.id)
        await session.commit()


asyncio.run(main())
PY
  )
}

start_backend_process() {
  if is_pid_running "${BACKEND_PID_FILE}"; then
    return 0
  fi

  : >"${BACKEND_LOG}"
  (
    cd "${BACKEND_SERVER_DIR}"
    nohup env \
      APP_ROOT="${JOINT_APP_ROOT}" \
      APP_PROFILE="${JOINT_APP_PROFILE}" \
      MODELS_PROFILE="${JOINT_MODELS_PROFILE}" \
      MODELS_YAML="${JOINT_MODELS_YAML}" \
      SCHEMAS_DIR="${JOINT_SCHEMAS_DIR}" \
      DATABASE_URL="${JOINT_DATABASE_URL}" \
      REDIS_ENABLED=1 \
      REDIS_URL="${JOINT_REDIS_URL}" \
      EDGE_MODE=behind_gateway \
      OTEL_ENABLED="${JOINT_WITH_OTEL}" \
      OTEL_SERVICE_NAME="${JOINT_BACKEND_OTEL_SERVICE_NAME}" \
      OTEL_EXPORTER_OTLP_ENDPOINT="${JOINT_OTEL_EXPORTER_OTLP_ENDPOINT}" \
      uv run python -m uvicorn llm_server.main:app \
        --host "${JOINT_BACKEND_HOST}" \
        --port "${JOINT_BACKEND_PORT}" \
      >"${BACKEND_LOG}" 2>&1 &
    echo $! >"${BACKEND_PID_FILE}"
  )
}

start_worker_process() {
  if is_pid_running "${WORKER_PID_FILE}"; then
    return 0
  fi

  : >"${WORKER_LOG}"
  (
    cd "${BACKEND_SERVER_DIR}"
    nohup env \
      APP_ROOT="${JOINT_APP_ROOT}" \
      APP_PROFILE="${JOINT_APP_PROFILE}" \
      MODELS_PROFILE="${JOINT_MODELS_PROFILE}" \
      MODELS_YAML="${JOINT_MODELS_YAML}" \
      SCHEMAS_DIR="${JOINT_SCHEMAS_DIR}" \
      DATABASE_URL="${JOINT_DATABASE_URL}" \
      REDIS_ENABLED=1 \
      REDIS_URL="${JOINT_REDIS_URL}" \
      EDGE_MODE=behind_gateway \
      OTEL_ENABLED="${JOINT_WITH_OTEL}" \
      OTEL_SERVICE_NAME="${JOINT_WORKER_OTEL_SERVICE_NAME}" \
      OTEL_EXPORTER_OTLP_ENDPOINT="${JOINT_OTEL_EXPORTER_OTLP_ENDPOINT}" \
      uv run python -m llm_server.worker.extract_jobs --poll-timeout-seconds 1 \
      >"${WORKER_LOG}" 2>&1 &
    echo $! >"${WORKER_PID_FILE}"
  )
}

start_gateway_process() {
  if is_pid_running "${GATEWAY_PID_FILE}"; then
    return 0
  fi

  : >"${GATEWAY_LOG}"
  (
    cd "${ISG_REPO_ROOT}"
    go build -o "${GATEWAY_BIN}" ./cmd/gateway
    nohup env \
      GATEWAY_LISTEN_ADDR="${JOINT_GATEWAY_HOST}:${JOINT_GATEWAY_PORT}" \
      GATEWAY_UPSTREAM_BASE_URL="$(backend_url)" \
      GATEWAY_REQUEST_TIMEOUT=30s \
      GATEWAY_ENABLE_METRICS=true \
      GATEWAY_ALLOW_EXTRACT=true \
      GATEWAY_ALLOW_EXTRACT_JOBS=true \
      GATEWAY_ALLOW_JOB_STATUS=true \
      GATEWAY_OTEL_ENABLED="${JOINT_WITH_OTEL}" \
      GATEWAY_OTEL_SERVICE_NAME="${JOINT_GATEWAY_OTEL_SERVICE_NAME}" \
      GATEWAY_OTEL_EXPORTER_OTLP_ENDPOINT="${JOINT_OTEL_EXPORTER_OTLP_ENDPOINT}" \
      "${GATEWAY_BIN}" \
      >"${GATEWAY_LOG}" 2>&1 &
    echo $! >"${GATEWAY_PID_FILE}"
  )
}

show_status_line() {
  local label="$1"
  local pid_file="$2"
  local health_url="${3:-}"
  local process_state="stopped"
  if is_pid_running "${pid_file}"; then
    process_state="running"
  fi
  if [[ -n "${health_url}" ]]; then
    if curl -fsS "${health_url}" >/dev/null 2>&1; then
      echo "${label}: ${process_state} (${health_url} healthy)"
    else
      echo "${label}: ${process_state} (${health_url} unhealthy)"
    fi
  else
    echo "${label}: ${process_state}"
  fi
}

cmd_preflight() {
  need_cmd bash
  need_cmd curl
  need_cmd docker
  need_cmd go
  need_cmd nohup
  need_cmd python3
  need_cmd uv
  ensure_docker_ready

  need_dir "${ISG_REPO_ROOT}"
  need_file "${ISG_REPO_ROOT}/go.mod"
  need_file "${ISG_REPO_ROOT}/cmd/gateway/main.go"
  need_file "${COMPOSE_FILE}"
  need_file "${JOINT_MODELS_YAML}"
  need_file "${JOINT_SCHEMAS_DIR}/sroie_receipt_v1.json"
  need_file "${ISG_REPO_ROOT}/proof/generate_llm_extraction_platform_observability_pack.sh"

  check_port_available "127.0.0.1" "${JOINT_BACKEND_PORT}"
  check_port_available "127.0.0.1" "${JOINT_GATEWAY_PORT}"
  check_port_available "127.0.0.1" "${JOINT_PG_HOST_PORT}"
  check_port_available "127.0.0.1" "${JOINT_REDIS_HOST_PORT}"
  if [[ "${JOINT_WITH_OBS}" == "1" ]]; then
    check_port_available "127.0.0.1" "${JOINT_PROM_HOST_PORT}"
    check_port_available "127.0.0.1" "${JOINT_GRAFANA_PORT}"
  fi
  if [[ "${JOINT_WITH_OTEL}" == "1" ]]; then
    check_port_available "127.0.0.1" "${JOINT_OTEL_COLLECTOR_PORT}"
    check_port_available "127.0.0.1" "${JOINT_OTEL_COLLECTOR_HEALTH_PORT}"
    check_port_available "127.0.0.1" "${JOINT_JAEGER_PORT}"
  fi

  if [[ "${JOINT_RUN_TESTS}" == "1" ]]; then
    (cd "${ISG_REPO_ROOT}" && go test ./...)
  else
    (cd "${ISG_REPO_ROOT}" && go test ./cmd/gateway ./internal/config)
  fi

  echo "Joint gateway preflight passed."
}

cmd_up() {
  cmd_preflight
  ensure_layout
  write_env_file

  compose_with_profiles up -d --remove-orphans

  wait_for_tcp "127.0.0.1" "${JOINT_PG_HOST_PORT}" 120
  wait_for_database_url "${JOINT_DATABASE_URL}" 120
  wait_for_tcp "127.0.0.1" "${JOINT_REDIS_HOST_PORT}" 120
  if [[ "${JOINT_WITH_OTEL}" == "1" ]]; then
    wait_for_tcp "127.0.0.1" "${JOINT_OTEL_COLLECTOR_PORT}" 120
    wait_for_url "http://127.0.0.1:${JOINT_JAEGER_PORT}"
  fi

  run_migrations
  seed_proof_keys
  start_backend_process
  start_worker_process
  start_gateway_process

  wait_for_url "$(backend_url)/healthz" 160
  wait_for_url "$(gateway_url)/healthz" 160

  echo "Joint gateway stack is up."
  echo "Backend: $(backend_url)"
  echo "Gateway: $(gateway_url)"
  echo "Prometheus: http://127.0.0.1:${JOINT_PROM_HOST_PORT} (if JOINT_WITH_OBS=1)"
  echo "Grafana: http://127.0.0.1:${JOINT_GRAFANA_PORT} (if JOINT_WITH_OBS=1)"
  echo "Jaeger: http://127.0.0.1:${JOINT_JAEGER_PORT} (if JOINT_WITH_OTEL=1)"
  echo "Env contract: ${ENV_FILE}"
}

cmd_down() {
  stop_pid_file "${GATEWAY_PID_FILE}"
  stop_pid_file "${WORKER_PID_FILE}"
  stop_pid_file "${BACKEND_PID_FILE}"
  if command -v docker >/dev/null 2>&1; then
    compose_with_profiles down --remove-orphans || true
  fi
  echo "Joint gateway stack is down."
}

cmd_restart() {
  cmd_down
  cmd_up
}

cmd_verify() {
  trap cmd_down EXIT
  cmd_up
  cmd_proof
}

cmd_status() {
  show_status_line "backend" "${BACKEND_PID_FILE}" "$(backend_url)/healthz"
  show_status_line "worker" "${WORKER_PID_FILE}"
  show_status_line "gateway" "${GATEWAY_PID_FILE}" "$(gateway_url)/healthz"
  if command -v docker >/dev/null 2>&1; then
    if docker info >/dev/null 2>&1; then
      echo "infra:"
      compose_with_profiles ps postgres_host redis_host prometheus_host grafana otel_collector_host jaeger || true
    else
      echo "infra: docker unavailable"
    fi
  fi
  echo "logs:"
  echo "  backend -> ${BACKEND_LOG}"
  echo "  worker  -> ${WORKER_LOG}"
  echo "  gateway -> ${GATEWAY_LOG}"
}

cmd_proof() {
  if ! is_pid_running "${BACKEND_PID_FILE}" \
    || ! is_pid_running "${WORKER_PID_FILE}" \
    || ! is_pid_running "${GATEWAY_PID_FILE}"; then
    echo "Joint stack is not fully running. Run: tools/joint/inference_gateway_stack.sh up" >&2
    exit 1
  fi

  wait_for_url "$(backend_url)/healthz"
  wait_for_url "$(gateway_url)/healthz"
  mkdir -p "${JOINT_ARTIFACT_DIR}"

  local jaeger_base_url=""
  if [[ "${JOINT_WITH_OTEL}" == "1" ]]; then
    jaeger_base_url="http://127.0.0.1:${JOINT_JAEGER_PORT}"
  fi

  env \
    LLM_EXTRACTION_PLATFORM_BASE_URL="$(backend_url)" \
    LLM_EXTRACTION_PLATFORM_API_KEY="${JOINT_PROOF_USER_KEY}" \
    LLM_EXTRACTION_PLATFORM_ADMIN_API_KEY="${JOINT_PROOF_ADMIN_KEY}" \
    GATEWAY_BASE_URL="$(gateway_url)" \
    GATEWAY_LOG_PATH="${GATEWAY_LOG}" \
    JAEGER_BASE_URL="${jaeger_base_url}" \
    OTEL_GATEWAY_SERVICE_NAME="${JOINT_GATEWAY_OTEL_SERVICE_NAME}" \
    OTEL_BACKEND_SERVICE_NAME="${JOINT_BACKEND_OTEL_SERVICE_NAME}" \
    OTEL_WORKER_SERVICE_NAME="${JOINT_WORKER_OTEL_SERVICE_NAME}" \
    "${ISG_REPO_ROOT}/proof/generate_llm_extraction_platform_observability_pack.sh" \
    "${JOINT_ARTIFACT_DIR}"

  cp "${BACKEND_LOG}" "${JOINT_ARTIFACT_DIR}/backend.log" || true
  cp "${WORKER_LOG}" "${JOINT_ARTIFACT_DIR}/worker.log" || true
  cp "${ENV_FILE}" "${JOINT_ARTIFACT_DIR}/joint-gateway.env" || true
  echo "Joint gateway proof artifacts: ${JOINT_ARTIFACT_DIR}"
}

main() {
  local cmd="${1:-}"
  case "${cmd}" in
    preflight) shift; cmd_preflight "$@" ;;
    up) shift; cmd_up "$@" ;;
    down) shift; cmd_down "$@" ;;
    restart) shift; cmd_restart "$@" ;;
    status) shift; cmd_status "$@" ;;
    proof) shift; cmd_proof "$@" ;;
    verify) shift; cmd_verify "$@" ;;
    ""|-h|--help|help) usage ;;
    *)
      echo "Unknown command: ${cmd}" >&2
      usage
      exit 2
      ;;
  esac
}

main "$@"
