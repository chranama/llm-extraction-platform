#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLMEP_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CAREER_ROOT="$(cd "${LLMEP_ROOT}/.." && pwd)"

: "${ISG_REPO_ROOT:=${CAREER_ROOT}/inference-serving-gateway}"

BACKEND_SERVER_DIR="${LLMEP_ROOT}/server"
COMPOSE_FILE="${LLMEP_ROOT}/deploy/compose/docker-compose.yml"
GATEWAY_COMPOSE_FILE="${LLMEP_ROOT}/deploy/compose/docker-compose.gateway.yml"
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
: "${JOINT_OBSERVABILITY_ARTIFACT_DIR:=${LLMEP_ROOT}/proof/artifacts/joint_gateway/observability_latest}"
: "${JOINT_EDGE_ARTIFACT_DIR:=${LLMEP_ROOT}/proof/artifacts/joint_gateway/edge_controls_latest}"
: "${JOINT_LLAMA_ARTIFACT_DIR:=${LLMEP_ROOT}/proof/artifacts/joint_gateway/llama_extract_latest}"
: "${JOINT_CONTAINER_ARTIFACT_DIR:=${LLMEP_ROOT}/proof/artifacts/joint_gateway/containerized_latest}"
: "${JOINT_CONTAINER_LLAMA_ARTIFACT_DIR:=${LLMEP_ROOT}/proof/artifacts/joint_gateway/containerized_llama_latest}"
: "${JOINT_RESILIENCE_ARTIFACT_DIR:=${LLMEP_ROOT}/proof/artifacts/joint_gateway/resilience_latest}"
: "${JOINT_KIND_ARTIFACT_DIR:=${LLMEP_ROOT}/proof/artifacts/joint_gateway/kind_live_latest}"
: "${JOINT_OTEL_EXPORTER_OTLP_ENDPOINT:=http://127.0.0.1:${JOINT_OTEL_COLLECTOR_PORT}/v1/traces}"
: "${JOINT_BACKEND_OTEL_SERVICE_NAME:=llm-extraction-platform}"
: "${JOINT_WORKER_OTEL_SERVICE_NAME:=llm-extraction-platform-worker}"
: "${JOINT_GATEWAY_OTEL_SERVICE_NAME:=inference-serving-gateway}"
: "${JOINT_GATEWAY_REQUEST_TIMEOUT:=30s}"
: "${JOINT_GATEWAY_ENABLE_METRICS:=true}"
: "${JOINT_GATEWAY_ALLOW_EXTRACT:=true}"
: "${JOINT_GATEWAY_ALLOW_EXTRACT_JOBS:=true}"
: "${JOINT_GATEWAY_ALLOW_JOB_STATUS:=true}"
: "${JOINT_GATEWAY_MAX_BODY_BYTES:=1048576}"
: "${JOINT_GATEWAY_CONCURRENCY_LIMIT:=64}"
: "${JOINT_GATEWAY_RATE_LIMIT_PER_SECOND:=0}"
: "${JOINT_GATEWAY_RATE_LIMIT_BURST:=1}"

: "${JOINT_LLAMA_ENV_FILE:=${LLMEP_ROOT}/.env.docker}"
: "${JOINT_LLAMA_EFFECTIVE_ENV_FILE:=${RUNTIME_DIR}/llama-compose.env}"
: "${JOINT_LLAMA_PROJECT_NAME:=llmep_joint_llama}"
: "${JOINT_LLAMA_API_PORT:=18190}"
: "${JOINT_LLAMA_GATEWAY_PORT:=18191}"
: "${JOINT_LLAMA_PUBLISH_PORT:=18192}"
: "${JOINT_LLAMA_API_KEY:=}"
: "${JOINT_LLAMA_ADMIN_API_KEY:=}"

: "${JOINT_CONTAINER_PROJECT_NAME:=llmep_joint_containerized}"
: "${JOINT_CONTAINER_API_PORT:=18290}"
: "${JOINT_CONTAINER_GATEWAY_PORT:=18291}"

: "${JOINT_CONTAINER_LLAMA_ENV_FILE:=${JOINT_LLAMA_ENV_FILE}}"
: "${JOINT_CONTAINER_LLAMA_EFFECTIVE_ENV_FILE:=${RUNTIME_DIR}/containerized-llama-compose.env}"
: "${JOINT_CONTAINER_LLAMA_PROJECT_NAME:=llmep_joint_containerized_llama}"
: "${JOINT_CONTAINER_LLAMA_API_PORT:=18390}"
: "${JOINT_CONTAINER_LLAMA_GATEWAY_PORT:=18391}"
: "${JOINT_CONTAINER_LLAMA_PUBLISH_PORT:=18392}"
: "${JOINT_CONTAINER_LLAMA_API_KEY:=}"
: "${JOINT_CONTAINER_LLAMA_ADMIN_API_KEY:=}"

: "${JOINT_RESILIENCE_PROJECT_NAME:=llmep_joint_resilience}"
: "${JOINT_RESILIENCE_API_PORT:=18490}"
: "${JOINT_RESILIENCE_GATEWAY_PORT:=18491}"
: "${JOINT_RESILIENCE_GATEWAY_TIMEOUT:=1s}"

: "${JOINT_KIND_CLUSTER:=llm}"
: "${JOINT_KIND_WORKFLOW:=live}"
: "${JOINT_KIND_ENV_FILE:=${LLMEP_ROOT}/.env.docker}"

usage() {
  cat <<'EOF'
Usage:
  tools/joint/inference_gateway_stack.sh kind-up
  tools/joint/inference_gateway_stack.sh kind-status
  tools/joint/inference_gateway_stack.sh kind-smoke
  tools/joint/inference_gateway_stack.sh kind-down
  tools/joint/inference_gateway_stack.sh preflight
  tools/joint/inference_gateway_stack.sh up
  tools/joint/inference_gateway_stack.sh status
  tools/joint/inference_gateway_stack.sh proof
  tools/joint/inference_gateway_stack.sh down
  tools/joint/inference_gateway_stack.sh restart
  tools/joint/inference_gateway_stack.sh verify
  tools/joint/inference_gateway_stack.sh verify-observability
  tools/joint/inference_gateway_stack.sh verify-edge-controls
  tools/joint/inference_gateway_stack.sh verify-llama
  tools/joint/inference_gateway_stack.sh verify-containerized
  tools/joint/inference_gateway_stack.sh verify-containerized-llama
  tools/joint/inference_gateway_stack.sh verify-resilience
  tools/joint/inference_gateway_stack.sh verify-kind

Promoted kind shape:
  - LLMEP API server, worker, llama-server, gateway, OTel, and Jaeger in kind
  - live CPU-only llama.cpp model runtime with host-mounted GGUF model
  - no proof artifacts from kind-smoke

Host evidence shape:
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
  JOINT_LLAMA_ENV_FILE=.env.docker
  JOINT_KIND_ENV_FILE=.env.docker
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

clear_artifact_dir() {
  local path="$1"
  mkdir -p "${path}"
  find "${path}" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
}

redact_artifact_api_keys() {
  local path="$1"
  [[ -d "${path}" ]] || return 0
  python3 - <<'PY' "${path}"
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])


def redact(value):
    changed = False
    if isinstance(value, dict):
        for key, item in list(value.items()):
            if key.lower() == "api_key" and item:
                value[key] = "<redacted>"
                changed = True
            else:
                next_value, next_changed = redact(item)
                if next_changed:
                    value[key] = next_value
                    changed = True
    elif isinstance(value, list):
        for index, item in enumerate(value):
            next_value, next_changed = redact(item)
            if next_changed:
                value[index] = next_value
                changed = True
    return value, changed


for path in root.rglob("*.json"):
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        continue
    payload, changed = redact(payload)
    if changed:
        path.write_text(json.dumps(payload, indent=2) + "\n")
PY
}

normalize_artifact_headers() {
  local path="$1"
  [[ -d "${path}" ]] || return 0
  python3 - <<'PY' "${path}"
import sys
from pathlib import Path

root = Path(sys.argv[1])
for path in root.rglob("*.headers"):
    text = path.read_bytes().decode("utf-8", errors="replace")
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]
    while lines and not lines[-1]:
        lines.pop()
    path.write_text("\n".join(lines) + "\n")
PY
}

normalize_artifact_text_files() {
  local path="$1"
  [[ -d "${path}" ]] || return 0
  python3 - <<'PY' "${path}"
import sys
from pathlib import Path

root = Path(sys.argv[1])
for path in root.rglob("*"):
    if not path.is_file():
        continue
    if path.suffix not in {".log", ".txt"}:
        continue
    text = path.read_bytes().decode("utf-8", errors="replace")
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]
    while lines and not lines[-1]:
        lines.pop()
    path.write_text("\n".join(lines) + "\n")
PY
}

clean_artifacts() {
  local path="$1"
  redact_artifact_api_keys "${path}"
  normalize_artifact_headers "${path}"
  normalize_artifact_text_files "${path}"
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

compose_containerized() {
  env \
    API_KEY="${JOINT_PROOF_USER_KEY}" \
    ADMIN_API_KEY="${JOINT_PROOF_ADMIN_KEY}" \
    API_PORT="${JOINT_CONTAINER_API_PORT}" \
    JOINT_GATEWAY_PORT="${JOINT_CONTAINER_GATEWAY_PORT}" \
    ISG_REPO_ROOT="${ISG_REPO_ROOT}" \
    APP_PROFILE="docker" \
    MODELS_PROFILE="test" \
    EDGE_MODE="behind_gateway" \
    REDIS_ENABLED="1" \
    DATABASE_URL="postgresql+asyncpg://llm:llm@postgres:5432/llm" \
    REDIS_URL="redis://redis:6379/0" \
    MODELS_YAML="/app/config/models.yaml" \
    SCHEMAS_DIR="/app/schemas/model_output" \
    POLICY_DECISION_PATH="/app/policy_out/latest.json" \
    CONTAINER_MEMORY_BYTES="4294967296" \
    OTEL_ENABLED="0" \
    COMPOSE_PROJECT_NAME="${JOINT_CONTAINER_PROJECT_NAME}" \
    docker compose \
      -f "${COMPOSE_FILE}" \
      -f "${GATEWAY_COMPOSE_FILE}" \
      --profile infra \
      --profile server \
      --profile joint-worker \
      --profile gateway \
      "$@"
}

compose_containerized_llama() {
  local api_key admin_key
  api_key="$(env_file_value "${JOINT_CONTAINER_LLAMA_EFFECTIVE_ENV_FILE}" "API_KEY")"
  admin_key="$(env_file_value "${JOINT_CONTAINER_LLAMA_EFFECTIVE_ENV_FILE}" "ADMIN_API_KEY")"
  env \
    API_KEY="${api_key}" \
    ADMIN_API_KEY="${admin_key}" \
    API_PORT="${JOINT_CONTAINER_LLAMA_API_PORT}" \
    LLAMA_PUBLISH_PORT="${JOINT_CONTAINER_LLAMA_PUBLISH_PORT}" \
    JOINT_GATEWAY_PORT="${JOINT_CONTAINER_LLAMA_GATEWAY_PORT}" \
    ISG_REPO_ROOT="${ISG_REPO_ROOT}" \
    APP_PROFILE="docker" \
    MODELS_PROFILE="compose-extract" \
    EDGE_MODE="behind_gateway" \
    REDIS_ENABLED="1" \
    DATABASE_URL="postgresql+asyncpg://llm:llm@postgres:5432/llm" \
    REDIS_URL="redis://redis:6379/0" \
    MODELS_YAML="/app/config/models.yaml" \
    SCHEMAS_DIR="/app/schemas/model_output" \
    LLAMA_SERVER_URL="http://llama_server:8080" \
    POLICY_DECISION_PATH="/app/policy_out/local_extract_allow.json" \
    CONTAINER_MEMORY_BYTES="4294967296" \
    OTEL_ENABLED="0" \
    GATEWAY_UPSTREAM_BASE_URL="http://server_llama:8000" \
    COMPOSE_PROJECT_NAME="${JOINT_CONTAINER_LLAMA_PROJECT_NAME}" \
    docker compose \
      --env-file "${JOINT_CONTAINER_LLAMA_EFFECTIVE_ENV_FILE}" \
      -f "${COMPOSE_FILE}" \
      -f "${GATEWAY_COMPOSE_FILE}" \
      --profile infra \
      --profile llama \
      --profile server-llama \
      --profile worker-llama \
      --profile gateway-llama \
      "$@"
}

compose_resilience() {
  env \
    API_KEY="${JOINT_PROOF_USER_KEY}" \
    ADMIN_API_KEY="${JOINT_PROOF_ADMIN_KEY}" \
    API_PORT="${JOINT_RESILIENCE_API_PORT}" \
    JOINT_GATEWAY_PORT="${JOINT_RESILIENCE_GATEWAY_PORT}" \
    ISG_REPO_ROOT="${ISG_REPO_ROOT}" \
    APP_PROFILE="docker" \
    MODELS_PROFILE="test" \
    EDGE_MODE="behind_gateway" \
    REDIS_ENABLED="1" \
    DATABASE_URL="postgresql+asyncpg://llm:llm@postgres:5432/llm" \
    REDIS_URL="redis://redis:6379/0" \
    MODELS_YAML="/app/config/models.yaml" \
    SCHEMAS_DIR="/app/schemas/model_output" \
    POLICY_DECISION_PATH="/app/policy_out/latest.json" \
    CONTAINER_MEMORY_BYTES="4294967296" \
    OTEL_ENABLED="0" \
    GATEWAY_REQUEST_TIMEOUT="${JOINT_RESILIENCE_GATEWAY_TIMEOUT}" \
    COMPOSE_PROJECT_NAME="${JOINT_RESILIENCE_PROJECT_NAME}" \
    docker compose \
      -f "${COMPOSE_FILE}" \
      -f "${GATEWAY_COMPOSE_FILE}" \
      --profile infra \
      --profile server \
      --profile joint-worker \
      --profile gateway \
      "$@"
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
      GATEWAY_REQUEST_TIMEOUT="${JOINT_GATEWAY_REQUEST_TIMEOUT}" \
      GATEWAY_ENABLE_METRICS="${JOINT_GATEWAY_ENABLE_METRICS}" \
      GATEWAY_ALLOW_EXTRACT="${JOINT_GATEWAY_ALLOW_EXTRACT}" \
      GATEWAY_ALLOW_EXTRACT_JOBS="${JOINT_GATEWAY_ALLOW_EXTRACT_JOBS}" \
      GATEWAY_ALLOW_JOB_STATUS="${JOINT_GATEWAY_ALLOW_JOB_STATUS}" \
      GATEWAY_MAX_BODY_BYTES="${JOINT_GATEWAY_MAX_BODY_BYTES}" \
      GATEWAY_CONCURRENCY_LIMIT="${JOINT_GATEWAY_CONCURRENCY_LIMIT}" \
      GATEWAY_RATE_LIMIT_PER_SECOND="${JOINT_GATEWAY_RATE_LIMIT_PER_SECOND}" \
      GATEWAY_RATE_LIMIT_BURST="${JOINT_GATEWAY_RATE_LIMIT_BURST}" \
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
  clean_artifacts "${JOINT_ARTIFACT_DIR}"

  cp "${BACKEND_LOG}" "${JOINT_ARTIFACT_DIR}/backend.log" || true
  cp "${WORKER_LOG}" "${JOINT_ARTIFACT_DIR}/worker.log" || true
  cp "${ENV_FILE}" "${JOINT_ARTIFACT_DIR}/joint-gateway.env" || true
  echo "Joint gateway proof artifacts: ${JOINT_ARTIFACT_DIR}"
}

capture_request() {
  local artifact_dir="$1"
  local name="$2"
  local method="$3"
  local url="$4"
  local api_key="$5"
  local request_body="${6:-}"
  local request_file="${7:-}"
  local request_id="${8:-}"
  local trace_id="${9:-}"

  local headers="${artifact_dir}/${name}.headers"
  local body="${artifact_dir}/${name}.body.json"
  local meta="${artifact_dir}/${name}.meta.json"
  local status
  local -a args=(curl -sS -D "${headers}" -o "${body}" -w "%{http_code}" -X "${method}")

  if [[ -n "${api_key}" ]]; then
    args+=(-H "X-API-Key: ${api_key}")
  fi
  if [[ -n "${request_id}" ]]; then
    args+=(-H "X-Request-ID: ${request_id}")
  fi
  if [[ -n "${trace_id}" ]]; then
    args+=(-H "X-Trace-ID: ${trace_id}")
  fi
  if [[ -n "${request_body}" || -n "${request_file}" ]]; then
    args+=(-H "Content-Type: application/json")
  fi
  if [[ -n "${request_file}" ]]; then
    args+=(--data-binary "@${request_file}")
  elif [[ -n "${request_body}" ]]; then
    args+=(--data "${request_body}")
  fi
  args+=("${url}")

  status="$("${args[@]}" || true)"
  python3 - <<'PY' "${meta}" "${name}" "${method}" "${url}" "${status}"
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "name": sys.argv[2],
    "method": sys.argv[3],
    "url": sys.argv[4],
    "status_code": int(sys.argv[5] or "0"),
}
path.write_text(json.dumps(payload, indent=2) + "\n")
PY
}

json_field() {
  local path="$1"
  local field="$2"
  python3 - <<'PY' "${path}" "${field}"
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
field = sys.argv[2]
try:
    value = json.loads(path.read_text())
except Exception:
    print("")
    raise SystemExit(0)
for part in field.split("."):
    if isinstance(value, dict):
        value = value.get(part)
    else:
        value = None
        break
if value is None:
    print("")
elif isinstance(value, (dict, list)):
    print(json.dumps(value))
else:
    print(value)
PY
}

write_edge_controls_manifest() {
  local artifact_dir="$1"
  python3 - <<'PY' "${artifact_dir}"
import json
import sys
from pathlib import Path

artifact_dir = Path(sys.argv[1])


def meta(name: str) -> dict:
    return json.loads((artifact_dir / f"{name}.meta.json").read_text())


def body(name: str) -> dict:
    path = artifact_dir / f"{name}.body.json"
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {"raw": path.read_text()}


def error_code(name: str) -> str:
    payload = body(name)
    error = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error, dict):
        return str(error.get("code") or "")
    return ""


metrics = {
    path.name: path.read_text()
    for path in artifact_dir.glob("*.metrics.txt")
}

checks = {
    "readyz_passes": meta("base_readyz")["status_code"] == 200,
    "allowed_extract_succeeds": meta("base_extract_allowed")["status_code"] == 200,
    "invalid_api_key_reaches_backend_auth": meta("base_invalid_api_key")["status_code"] == 401,
    "unsupported_generate_is_gateway_owned": (
        meta("base_unsupported_generate")["status_code"] == 404
        and error_code("base_unsupported_generate") == "unsupported_route"
    ),
    "extract_route_disabled_by_gateway": (
        meta("disabled_extract")["status_code"] == 403
        and error_code("disabled_extract") == "route_not_allowed"
    ),
    "extract_jobs_route_disabled_by_gateway": (
        meta("disabled_extract_jobs")["status_code"] == 403
        and error_code("disabled_extract_jobs") == "route_not_allowed"
    ),
    "oversized_extract_rejected_by_gateway": (
        meta("oversized_extract")["status_code"] == 413
        and error_code("oversized_extract") == "request_too_large"
    ),
    "gateway_metrics_captured": any("gateway_requests_total" in text for text in metrics.values()),
}

manifest = {
    "mode": "joint_edge_controls",
    "checks": checks,
    "artifacts": {
        "base_readyz": "base_readyz.body.json",
        "base_extract_allowed": "base_extract_allowed.body.json",
        "base_invalid_api_key": "base_invalid_api_key.body.json",
        "base_unsupported_generate": "base_unsupported_generate.body.json",
        "disabled_extract": "disabled_extract.body.json",
        "disabled_extract_jobs": "disabled_extract_jobs.body.json",
        "oversized_extract": "oversized_extract.body.json",
        "base_gateway_metrics": "base_gateway.metrics.txt",
        "disabled_gateway_metrics": "disabled_gateway.metrics.txt",
        "oversized_gateway_metrics": "oversized_gateway.metrics.txt",
    },
    "case_status_codes": {
        path.stem.removesuffix(".meta"): json.loads(path.read_text()).get("status_code")
        for path in sorted(artifact_dir.glob("*.meta.json"))
    },
}
(artifact_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

summary = ["# Joint Edge Controls", ""]
summary.append("Checks:")
for name, ok in checks.items():
    summary.append(f"- {name}: {'pass' if ok else 'fail'}")
summary.append("")
summary.append("This proof uses the deterministic backend and isolates gateway-owned behavior.")
(artifact_dir / "summary.md").write_text("\n".join(summary) + "\n")

failed = [name for name, ok in checks.items() if not ok]
if failed:
    raise SystemExit(f"edge control checks failed: {failed}")
PY
}

cmd_proof_edge_controls() {
  local artifact_dir="${JOINT_EDGE_ARTIFACT_DIR}"
  clear_artifact_dir "${artifact_dir}"
  local payload='{"schema_id":"sroie_receipt_v1","text":"Vendor: ACME\nTotal: 10.00","cache":false,"repair":true}'

  wait_for_url "$(gateway_url)/healthz"
  capture_request "${artifact_dir}" "base_readyz" "GET" "$(gateway_url)/readyz" "" ""
  capture_request "${artifact_dir}" "base_extract_allowed" "POST" "$(gateway_url)/v1/extract" "${JOINT_PROOF_USER_KEY}" "${payload}"
  capture_request "${artifact_dir}" "base_invalid_api_key" "POST" "$(gateway_url)/v1/extract" "invalid-joint-key" "${payload}"
  capture_request "${artifact_dir}" "base_unsupported_generate" "POST" "$(gateway_url)/v1/generate" "${JOINT_PROOF_USER_KEY}" '{"prompt":"test"}'
  curl -fsS "$(gateway_url)/metrics" >"${artifact_dir}/base_gateway.metrics.txt"
  cp "${GATEWAY_LOG}" "${artifact_dir}/base_gateway.log" || true

  stop_pid_file "${GATEWAY_PID_FILE}"
  JOINT_GATEWAY_ALLOW_EXTRACT=false
  JOINT_GATEWAY_ALLOW_EXTRACT_JOBS=false
  JOINT_GATEWAY_ALLOW_JOB_STATUS=true
  start_gateway_process
  wait_for_url "$(gateway_url)/healthz"
  capture_request "${artifact_dir}" "disabled_extract" "POST" "$(gateway_url)/v1/extract" "${JOINT_PROOF_USER_KEY}" "${payload}"
  capture_request "${artifact_dir}" "disabled_extract_jobs" "POST" "$(gateway_url)/v1/extract/jobs" "${JOINT_PROOF_USER_KEY}" "${payload}"
  curl -fsS "$(gateway_url)/metrics" >"${artifact_dir}/disabled_gateway.metrics.txt"
  cp "${GATEWAY_LOG}" "${artifact_dir}/disabled_gateway.log" || true

  stop_pid_file "${GATEWAY_PID_FILE}"
  JOINT_GATEWAY_ALLOW_EXTRACT=true
  JOINT_GATEWAY_ALLOW_EXTRACT_JOBS=true
  JOINT_GATEWAY_MAX_BODY_BYTES=64
  start_gateway_process
  wait_for_url "$(gateway_url)/healthz"
  local large_payload="${artifact_dir}/oversized_request.json"
  python3 - <<'PY' "${large_payload}"
import json
import sys
from pathlib import Path

payload = {
    "schema_id": "sroie_receipt_v1",
    "text": "A" * 512,
    "cache": False,
    "repair": True,
}
Path(sys.argv[1]).write_text(json.dumps(payload))
PY
  capture_request "${artifact_dir}" "oversized_extract" "POST" "$(gateway_url)/v1/extract" "${JOINT_PROOF_USER_KEY}" "" "${large_payload}"
  curl -fsS "$(gateway_url)/metrics" >"${artifact_dir}/oversized_gateway.metrics.txt"
  cp "${GATEWAY_LOG}" "${artifact_dir}/oversized_gateway.log" || true

  write_edge_controls_manifest "${artifact_dir}"
  clean_artifacts "${artifact_dir}"
  echo "Joint edge-control artifacts: ${artifact_dir}"
}

cmd_verify_observability() {
  JOINT_ARTIFACT_DIR="${JOINT_OBSERVABILITY_ARTIFACT_DIR}"
  trap cmd_down EXIT
  cmd_up
  cmd_proof
}

cmd_verify_edge_controls() {
  JOINT_ARTIFACT_DIR="${JOINT_EDGE_ARTIFACT_DIR}"
  JOINT_WITH_OBS=0
  JOINT_WITH_OTEL=0
  trap cmd_down EXIT
  cmd_up
  cmd_proof_edge_controls
}

env_file_value() {
  local path="$1"
  local key="$2"
  python3 - <<'PY' "${path}" "${key}"
import sys
from pathlib import Path

path = Path(sys.argv[1])
key = sys.argv[2]
value = ""
if path.exists():
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip() == key:
            value = v.strip().strip('"').strip("'")
print(value)
PY
}

write_llama_env_file() {
  local source_file="$1"
  local target_file="$2"
  local api_port="$3"
  local llama_port="$4"
  local api_key_override="$5"
  local admin_key_override="$6"
  python3 - <<'PY' \
    "${source_file}" \
    "${target_file}" \
    "${api_port}" \
    "${llama_port}" \
    "${api_key_override}" \
    "${admin_key_override}"
import sys
from pathlib import Path

source = Path(sys.argv[1]).expanduser()
target = Path(sys.argv[2])
api_port = sys.argv[3]
llama_port = sys.argv[4]
api_key_override = sys.argv[5]
admin_key_override = sys.argv[6]

if not source.exists():
    raise SystemExit(f"missing env file: {source}")

env: dict[str, str] = {}
for raw in source.read_text().splitlines():
    line = raw.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    key, value = line.split("=", 1)
    env[key.strip()] = value.strip().strip('"').strip("'")

if api_key_override:
    env["API_KEY"] = api_key_override
if admin_key_override:
    env["ADMIN_API_KEY"] = admin_key_override

env["API_PORT"] = api_port
env["LLAMA_PUBLISH_PORT"] = llama_port
env["EDGE_MODE"] = "behind_gateway"
env["OTEL_ENABLED"] = "0"
env["POLICY_DECISION_PATH"] = "/app/policy_out/local_extract_allow.json"
env.setdefault("ADMIN_API_KEY", env.get("API_KEY", ""))

target.parent.mkdir(parents=True, exist_ok=True)
target.write_text("\n".join(f"{k}={v}" for k, v in sorted(env.items())) + "\n")
PY
}

write_llama_effective_env() {
  write_llama_env_file \
    "${JOINT_LLAMA_ENV_FILE}" \
    "${JOINT_LLAMA_EFFECTIVE_ENV_FILE}" \
    "${JOINT_LLAMA_API_PORT}" \
    "${JOINT_LLAMA_PUBLISH_PORT}" \
    "${JOINT_LLAMA_API_KEY}" \
    "${JOINT_LLAMA_ADMIN_API_KEY}"
}

write_containerized_llama_effective_env() {
  write_llama_env_file \
    "${JOINT_CONTAINER_LLAMA_ENV_FILE}" \
    "${JOINT_CONTAINER_LLAMA_EFFECTIVE_ENV_FILE}" \
    "${JOINT_CONTAINER_LLAMA_API_PORT}" \
    "${JOINT_CONTAINER_LLAMA_PUBLISH_PORT}" \
    "${JOINT_CONTAINER_LLAMA_API_KEY}" \
    "${JOINT_CONTAINER_LLAMA_ADMIN_API_KEY}"
}

cmd_stop_llama_compose() {
  if [[ -f "${JOINT_LLAMA_EFFECTIVE_ENV_FILE}" ]]; then
    uv run llmctl \
      --project-name "${JOINT_LLAMA_PROJECT_NAME}" \
      --env-override-file "${JOINT_LLAMA_EFFECTIVE_ENV_FILE}" \
      stop --volumes || true
  fi
}

cmd_verify_llama() {
  need_cmd docker
  need_cmd uv
  need_cmd go
  need_cmd curl
  ensure_docker_ready
  need_file "${JOINT_LLAMA_ENV_FILE}"

  clear_artifact_dir "${JOINT_LLAMA_ARTIFACT_DIR}"
  write_llama_effective_env

  local api_key admin_key
  api_key="$(env_file_value "${JOINT_LLAMA_EFFECTIVE_ENV_FILE}" "API_KEY")"
  admin_key="$(env_file_value "${JOINT_LLAMA_EFFECTIVE_ENV_FILE}" "ADMIN_API_KEY")"
  if [[ -z "${api_key}" || -z "${admin_key}" ]]; then
    echo "Live llama joint workflow requires API_KEY and ADMIN_API_KEY in ${JOINT_LLAMA_EFFECTIVE_ENV_FILE}" >&2
    exit 1
  fi

  cleanup() {
    stop_pid_file "${GATEWAY_PID_FILE}"
    cmd_stop_llama_compose
  }
  trap cleanup EXIT

  uv run llmctl \
    --project-name "${JOINT_LLAMA_PROJECT_NAME}" \
    --env-override-file "${JOINT_LLAMA_EFFECTIVE_ENV_FILE}" \
    --api-port "${JOINT_LLAMA_API_PORT}" \
    compose-extract \
    >"${JOINT_LLAMA_ARTIFACT_DIR}/llmctl_compose_extract.log" 2>&1

  JOINT_BACKEND_PORT="${JOINT_LLAMA_API_PORT}"
  JOINT_GATEWAY_PORT="${JOINT_LLAMA_GATEWAY_PORT}"
  JOINT_WITH_OTEL=0
  JOINT_GATEWAY_MAX_BODY_BYTES=1048576
  JOINT_GATEWAY_ALLOW_EXTRACT=true
  JOINT_GATEWAY_ALLOW_EXTRACT_JOBS=true
  JOINT_GATEWAY_ALLOW_JOB_STATUS=true
  start_gateway_process
  wait_for_url "$(gateway_url)/healthz"
  wait_for_url "$(gateway_url)/readyz" 160

  env \
    LLM_EXTRACTION_PLATFORM_BASE_URL="$(backend_url)" \
    LLM_EXTRACTION_PLATFORM_API_KEY="${api_key}" \
    LLM_EXTRACTION_PLATFORM_ADMIN_API_KEY="${admin_key}" \
    GATEWAY_BASE_URL="$(gateway_url)" \
    GATEWAY_LOG_PATH="${GATEWAY_LOG}" \
    "${ISG_REPO_ROOT}/proof/generate_llm_extraction_platform_observability_pack.sh" \
    "${JOINT_LLAMA_ARTIFACT_DIR}"
  clean_artifacts "${JOINT_LLAMA_ARTIFACT_DIR}"

  docker compose \
    --env-file "${JOINT_LLAMA_EFFECTIVE_ENV_FILE}" \
    -f "${COMPOSE_FILE}" \
    -p "${JOINT_LLAMA_PROJECT_NAME}" \
    ps >"${JOINT_LLAMA_ARTIFACT_DIR}/compose.ps.txt" 2>&1 || true
  docker compose \
    --env-file "${JOINT_LLAMA_EFFECTIVE_ENV_FILE}" \
    -f "${COMPOSE_FILE}" \
    -p "${JOINT_LLAMA_PROJECT_NAME}" \
    logs --no-color --tail=200 server_llama llama_server worker_llama \
    >"${JOINT_LLAMA_ARTIFACT_DIR}/compose.logs.txt" 2>&1 || true
  cp "${GATEWAY_LOG}" "${JOINT_LLAMA_ARTIFACT_DIR}/gateway.log" || true

  python3 - <<'PY' "${JOINT_LLAMA_ARTIFACT_DIR}"
import json
import sys
from pathlib import Path

artifact_dir = Path(sys.argv[1])
manifest_path = artifact_dir / "manifest.json"
manifest = json.loads(manifest_path.read_text())
manifest["mode"] = "joint_llama_extract"
manifest["runtime"] = {
    "backend": "containerized llama.cpp",
    "gateway": "host-run Go gateway",
    "acceleration": "cpu",
}
manifest["checks"]["live_llama_extract_response_present"] = (artifact_dir / "extract.body.json").exists()
manifest["artifacts"]["llmctl_compose_extract_log"] = "llmctl_compose_extract.log"
manifest["artifacts"]["compose_ps"] = "compose.ps.txt"
manifest["artifacts"]["compose_logs"] = "compose.logs.txt"
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
PY
  echo "Joint live llama artifacts: ${JOINT_LLAMA_ARTIFACT_DIR}"
}

seed_containerized_keys() {
  local sql
  sql=$(
    cat <<EOF
INSERT INTO roles (name, created_at) VALUES ('admin', now())
ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name;
INSERT INTO roles (name, created_at) VALUES ('standard', now())
ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name;
INSERT INTO api_keys (key, active, quota_monthly, quota_used, created_at, role_id)
SELECT '${JOINT_PROOF_USER_KEY}', true, NULL, 0, now(), id FROM roles WHERE name = 'standard'
ON CONFLICT (key) DO UPDATE SET active = EXCLUDED.active, role_id = EXCLUDED.role_id;
INSERT INTO api_keys (key, active, quota_monthly, quota_used, created_at, role_id)
SELECT '${JOINT_PROOF_ADMIN_KEY}', true, NULL, 0, now(), id FROM roles WHERE name = 'admin'
ON CONFLICT (key) DO UPDATE SET active = EXCLUDED.active, role_id = EXCLUDED.role_id;
EOF
  )
  compose_containerized exec -T postgres psql -U llm -d llm -v ON_ERROR_STOP=1 -c "${sql}"
}

seed_resilience_keys() {
  local sql
  sql=$(
    cat <<EOF
INSERT INTO roles (name, created_at) VALUES ('admin', now())
ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name;
INSERT INTO roles (name, created_at) VALUES ('standard', now())
ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name;
INSERT INTO api_keys (key, active, quota_monthly, quota_used, created_at, role_id)
SELECT '${JOINT_PROOF_USER_KEY}', true, NULL, 0, now(), id FROM roles WHERE name = 'standard'
ON CONFLICT (key) DO UPDATE SET active = EXCLUDED.active, role_id = EXCLUDED.role_id;
INSERT INTO api_keys (key, active, quota_monthly, quota_used, created_at, role_id)
SELECT '${JOINT_PROOF_ADMIN_KEY}', true, NULL, 0, now(), id FROM roles WHERE name = 'admin'
ON CONFLICT (key) DO UPDATE SET active = EXCLUDED.active, role_id = EXCLUDED.role_id;
EOF
  )
  compose_resilience exec -T postgres psql -U llm -d llm -v ON_ERROR_STOP=1 -c "${sql}"
}

seed_containerized_llama_keys() {
  local api_key admin_key
  api_key="$(env_file_value "${JOINT_CONTAINER_LLAMA_EFFECTIVE_ENV_FILE}" "API_KEY")"
  admin_key="$(env_file_value "${JOINT_CONTAINER_LLAMA_EFFECTIVE_ENV_FILE}" "ADMIN_API_KEY")"
  if [[ -z "${api_key}" || -z "${admin_key}" ]]; then
    echo "Containerized llama joint workflow requires API_KEY and ADMIN_API_KEY in ${JOINT_CONTAINER_LLAMA_EFFECTIVE_ENV_FILE}" >&2
    exit 1
  fi
  compose_containerized_llama exec -T \
    -e PROOF_USER_KEY="${api_key}" \
    -e PROOF_ADMIN_KEY="${admin_key}" \
    server_llama python - <<'PY'
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
}

cmd_down_containerized() {
  if command -v docker >/dev/null 2>&1; then
    compose_containerized down --remove-orphans --volumes || true
  fi
}

cmd_down_resilience() {
  if command -v docker >/dev/null 2>&1; then
    compose_resilience down --remove-orphans --volumes || true
  fi
}

cmd_down_containerized_llama() {
  if command -v docker >/dev/null 2>&1 && [[ -f "${JOINT_CONTAINER_LLAMA_EFFECTIVE_ENV_FILE}" ]]; then
    compose_containerized_llama down --remove-orphans --volumes || true
  fi
}

cmd_verify_containerized() {
  need_cmd docker
  need_cmd curl
  ensure_docker_ready
  need_file "${GATEWAY_COMPOSE_FILE}"
  clear_artifact_dir "${JOINT_CONTAINER_ARTIFACT_DIR}"

  trap cmd_down_containerized EXIT
  compose_containerized up -d --build --remove-orphans postgres redis server
  wait_for_url "http://127.0.0.1:${JOINT_CONTAINER_API_PORT}/healthz" 180
  compose_containerized exec -T server python -m alembic -c /app/server/alembic.ini upgrade head
  seed_containerized_keys
  compose_containerized up -d --build --remove-orphans joint_worker gateway
  wait_for_url "http://127.0.0.1:${JOINT_CONTAINER_GATEWAY_PORT}/healthz" 180
  wait_for_url "http://127.0.0.1:${JOINT_CONTAINER_GATEWAY_PORT}/readyz" 180

  env \
    LLM_EXTRACTION_PLATFORM_BASE_URL="http://127.0.0.1:${JOINT_CONTAINER_API_PORT}" \
    LLM_EXTRACTION_PLATFORM_API_KEY="${JOINT_PROOF_USER_KEY}" \
    LLM_EXTRACTION_PLATFORM_ADMIN_API_KEY="${JOINT_PROOF_ADMIN_KEY}" \
    GATEWAY_BASE_URL="http://127.0.0.1:${JOINT_CONTAINER_GATEWAY_PORT}" \
    "${ISG_REPO_ROOT}/proof/generate_llm_extraction_platform_observability_pack.sh" \
    "${JOINT_CONTAINER_ARTIFACT_DIR}"
  clean_artifacts "${JOINT_CONTAINER_ARTIFACT_DIR}"

  compose_containerized ps >"${JOINT_CONTAINER_ARTIFACT_DIR}/compose.ps.txt" 2>&1 || true
  compose_containerized logs --no-color --tail=200 server joint_worker gateway \
    >"${JOINT_CONTAINER_ARTIFACT_DIR}/compose.logs.txt" 2>&1 || true
  python3 - <<'PY' "${JOINT_CONTAINER_ARTIFACT_DIR}"
import json
import sys
from pathlib import Path

artifact_dir = Path(sys.argv[1])
manifest_path = artifact_dir / "manifest.json"
manifest = json.loads(manifest_path.read_text())
manifest["mode"] = "joint_containerized_stack"
manifest["runtime"] = {
    "backend": "containerized LLMEP fake backend",
    "worker": "containerized LLMEP async worker",
    "gateway": "containerized inference-serving-gateway",
}
manifest["artifacts"]["compose_ps"] = "compose.ps.txt"
manifest["artifacts"]["compose_logs"] = "compose.logs.txt"
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
PY
  echo "Joint containerized artifacts: ${JOINT_CONTAINER_ARTIFACT_DIR}"
}

resilience_backend_url() {
  printf 'http://127.0.0.1:%s\n' "${JOINT_RESILIENCE_API_PORT}"
}

resilience_gateway_url() {
  printf 'http://127.0.0.1:%s\n' "${JOINT_RESILIENCE_GATEWAY_PORT}"
}

resilience_extract_payload() {
  python3 - <<'PY'
import json

print(
    json.dumps(
        {
            "schema_id": "sroie_receipt_v1",
            "text": "Vendor: ACME\nTotal: 10.00",
            "cache": False,
            "repair": True,
        }
    )
)
PY
}

capture_resilience_state() {
  local artifact_dir="$1"
  local phase="$2"
  compose_resilience ps >"${artifact_dir}/compose.ps.${phase}.txt" 2>&1 || true
}

capture_resilience_logs() {
  local artifact_dir="$1"
  compose_resilience logs --no-color --tail=200 server joint_worker gateway postgres redis \
    >"${artifact_dir}/compose.logs.txt" 2>&1 || true
}

capture_resilience_metrics() {
  local artifact_dir="$1"
  local phase="$2"
  curl -sS "$(resilience_gateway_url)/metrics" >"${artifact_dir}/gateway.metrics.${phase}.txt" 2>&1 || true
  curl -sS "$(resilience_backend_url)/metrics" >"${artifact_dir}/backend.metrics.${phase}.txt" 2>&1 || true
}

capture_resilience_trace() {
  local artifact_dir="$1"
  local name="$2"
  local trace_id="$3"
  capture_request \
    "${artifact_dir}" \
    "${name}" \
    "GET" \
    "$(resilience_backend_url)/v1/admin/traces/${trace_id}" \
    "${JOINT_PROOF_ADMIN_KEY}"
}

poll_resilience_job_until_terminal() {
  local artifact_dir="$1"
  local name="$2"
  local job_id="$3"
  local request_id="$4"
  local trace_id="$5"
  local attempts="${6:-80}"
  local status=""

  if [[ -z "${job_id}" ]]; then
    python3 - <<'PY' "${artifact_dir}" "${name}" "${request_id}" "${trace_id}"
import json
import sys
from pathlib import Path

artifact_dir = Path(sys.argv[1])
name = sys.argv[2]
request_id = sys.argv[3]
trace_id = sys.argv[4]
(artifact_dir / f"{name}.body.json").write_text(
    json.dumps(
        {
            "status": "missing_job_id",
            "request_id": request_id,
            "trace_id": trace_id,
        },
        indent=2,
    )
    + "\n"
)
(artifact_dir / f"{name}.headers").write_text("")
(artifact_dir / f"{name}.meta.json").write_text(
    json.dumps(
        {
            "name": name,
            "method": "GET",
            "url": "",
            "status_code": 0,
        },
        indent=2,
    )
    + "\n"
)
PY
    return 0
  fi

  for _ in $(seq 1 "${attempts}"); do
    capture_request \
      "${artifact_dir}" \
      "${name}" \
      "GET" \
      "$(resilience_gateway_url)/v1/extract/jobs/${job_id}" \
      "${JOINT_PROOF_USER_KEY}" \
      "" \
      "" \
      "${request_id}" \
      "${trace_id}"
    status="$(json_field "${artifact_dir}/${name}.body.json" "status")"
    if [[ "${status}" == "succeeded" || "${status}" == "failed" ]]; then
      break
    fi
    sleep 0.25
  done
}

wait_resilience_postgres() {
  for _ in $(seq 1 80); do
    if compose_resilience exec -T postgres pg_isready -U llm -d llm >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.25
  done
  echo "Timed out waiting for resilience Postgres" >&2
  return 1
}

wait_resilience_redis() {
  for _ in $(seq 1 80); do
    if compose_resilience exec -T redis redis-cli ping >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.25
  done
  echo "Timed out waiting for resilience Redis" >&2
  return 1
}

write_resilience_manifest() {
  local artifact_dir="$1"
  python3 - <<'PY' "${artifact_dir}" "${JOINT_RESILIENCE_GATEWAY_TIMEOUT}"
import json
import sys
from pathlib import Path

artifact_dir = Path(sys.argv[1])
gateway_timeout = sys.argv[2]


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def meta(name: str) -> dict:
    return read_json(artifact_dir / f"{name}.meta.json")


def body(name: str) -> dict:
    return read_json(artifact_dir / f"{name}.body.json")


def status(name: str) -> int:
    value = meta(name).get("status_code")
    return value if isinstance(value, int) else 0


def error_code(name: str) -> str:
    error = body(name).get("error")
    if isinstance(error, dict):
        return str(error.get("code") or "")
    return ""


def job_status(name: str) -> str:
    return str(body(name).get("status") or "")


def trace_completed(name: str) -> bool:
    return body(name).get("status") == "completed"


def text(name: str) -> str:
    path = artifact_dir / name
    return path.read_text(errors="replace") if path.exists() else ""


gateway_final_metrics = text("gateway.metrics.final.txt")
backend_final_metrics = text("backend.metrics.final.txt")

checks = {
    "baseline_sync_extract_succeeds": status("baseline_sync_extract") == 200,
    "baseline_async_extract_succeeds": (
        status("baseline_async_final") == 200
        and job_status("baseline_async_final") == "succeeded"
    ),
    "backend_timeout_is_bounded": (
        status("api_paused_extract") == 504
        and error_code("api_paused_extract") == "upstream_timeout"
    ),
    "backend_unavailable_is_bounded": (
        status("api_down_readyz") == 503
        and error_code("api_down_readyz") == "upstream_unavailable"
        and status("api_down_extract") == 503
        and error_code("api_down_extract") == "upstream_unavailable"
    ),
    "backend_recovery_succeeds": status("api_recovery_extract") == 200,
    "worker_failure_preserves_job_state": (
        status("worker_down_async_submit") == 202
        and status("worker_down_job_status") == 200
        and job_status("worker_down_job_status") in {"queued", "running"}
    ),
    "worker_recovery_completes_job": (
        status("worker_recovery_job_status") == 200
        and job_status("worker_recovery_job_status") == "succeeded"
    ),
    "redis_failure_is_observable": (
        status("redis_down_readyz") != 200
        or status("redis_down_async_submit") != 202
    ),
    "redis_recovery_succeeds": (
        status("redis_recovery_readyz") == 200
        and status("redis_recovery_extract") == 200
    ),
    "postgres_failure_is_observable": (
        status("postgres_down_readyz") != 200
        or status("postgres_down_extract") != 200
    ),
    "postgres_recovery_succeeds": (
        status("postgres_recovery_readyz") == 200
        and status("postgres_recovery_extract") == 200
    ),
    "gateway_metrics_capture_failures": (
        'result="timeout"' in gateway_final_metrics
        and 'result="unavailable"' in gateway_final_metrics
        and 'status="503"' in gateway_final_metrics
        and 'status="504"' in gateway_final_metrics
    ),
    "backend_metrics_capture_recovery": "llm_api_request_total" in backend_final_metrics,
    "traces_capture_recovery_flow": (
        trace_completed("api_recovery_trace")
        and trace_completed("worker_recovery_trace")
    ),
}

artifacts = {
    "baseline_sync_extract": "baseline_sync_extract.body.json",
    "baseline_async_submit": "baseline_async_submit.body.json",
    "baseline_async_final": "baseline_async_final.body.json",
    "api_paused_extract": "api_paused_extract.body.json",
    "api_down_readyz": "api_down_readyz.body.json",
    "api_down_extract": "api_down_extract.body.json",
    "api_recovery_extract": "api_recovery_extract.body.json",
    "worker_down_async_submit": "worker_down_async_submit.body.json",
    "worker_down_job_status": "worker_down_job_status.body.json",
    "worker_recovery_job_status": "worker_recovery_job_status.body.json",
    "redis_down_readyz": "redis_down_readyz.body.json",
    "redis_down_async_submit": "redis_down_async_submit.body.json",
    "redis_recovery_readyz": "redis_recovery_readyz.body.json",
    "redis_recovery_extract": "redis_recovery_extract.body.json",
    "postgres_down_readyz": "postgres_down_readyz.body.json",
    "postgres_down_extract": "postgres_down_extract.body.json",
    "postgres_recovery_readyz": "postgres_recovery_readyz.body.json",
    "postgres_recovery_extract": "postgres_recovery_extract.body.json",
    "gateway_metrics_final": "gateway.metrics.final.txt",
    "backend_metrics_final": "backend.metrics.final.txt",
    "compose_logs": "compose.logs.txt",
}

manifest = {
    "mode": "joint_resilience",
    "runtime": {
        "backend": "containerized LLMEP fake backend",
        "worker": "containerized LLMEP async worker",
        "gateway": "containerized inference-serving-gateway",
        "database": "containerized Postgres",
        "queue": "containerized Redis",
        "gateway_timeout": gateway_timeout,
    },
    "checks": checks,
    "artifacts": artifacts,
    "case_status_codes": {
        path.stem.removesuffix(".meta"): read_json(path).get("status_code")
        for path in sorted(artifact_dir.glob("*.meta.json"))
    },
}
(artifact_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

summary = ["# Joint Resilience Proof", ""]
summary.append(
    "This proof captures bounded degradation and operator-driven recovery for the local "
    "containerized LLMEP plus inference-serving-gateway stack."
)
summary.append("")
summary.append("Checks:")
for name, ok in checks.items():
    summary.append(f"- {name}: {'pass' if ok else 'fail'}")
summary.append("")
summary.append(
    "Scope: local resilience evidence only. This does not claim HA, autoscaling, "
    "zero downtime, cloud failover, or production incident response."
)
(artifact_dir / "summary.md").write_text("\n".join(summary) + "\n")

failed = [name for name, ok in checks.items() if not ok]
if failed:
    raise SystemExit(f"resilience checks failed: {failed}")
PY
}

cmd_verify_resilience() {
  need_cmd docker
  need_cmd curl
  ensure_docker_ready
  need_file "${GATEWAY_COMPOSE_FILE}"
  clear_artifact_dir "${JOINT_RESILIENCE_ARTIFACT_DIR}"

  local artifact_dir="${JOINT_RESILIENCE_ARTIFACT_DIR}"
  local payload
  payload="$(resilience_extract_payload)"

  trap cmd_down_resilience EXIT

  compose_resilience up -d --build --remove-orphans postgres redis server
  wait_for_url "$(resilience_backend_url)/healthz" 180
  compose_resilience exec -T server python -m alembic -c /app/server/alembic.ini upgrade head
  seed_resilience_keys
  compose_resilience up -d --build --remove-orphans joint_worker gateway
  wait_for_url "$(resilience_gateway_url)/healthz" 120
  wait_for_url "$(resilience_gateway_url)/readyz" 180
  capture_resilience_state "${artifact_dir}" "baseline"

  capture_request "${artifact_dir}" "baseline_readyz" "GET" "$(resilience_gateway_url)/readyz" ""
  capture_request \
    "${artifact_dir}" \
    "baseline_sync_extract" \
    "POST" \
    "$(resilience_gateway_url)/v1/extract" \
    "${JOINT_PROOF_USER_KEY}" \
    "${payload}" \
    "" \
    "resilience-baseline-sync-request" \
    "resilience-baseline-sync-trace"
  capture_request \
    "${artifact_dir}" \
    "baseline_async_submit" \
    "POST" \
    "$(resilience_gateway_url)/v1/extract/jobs" \
    "${JOINT_PROOF_USER_KEY}" \
    "${payload}" \
    "" \
    "resilience-baseline-async-submit-request" \
    "resilience-baseline-async-trace"
  local baseline_job_id
  baseline_job_id="$(json_field "${artifact_dir}/baseline_async_submit.body.json" "job_id")"
  poll_resilience_job_until_terminal \
    "${artifact_dir}" \
    "baseline_async_final" \
    "${baseline_job_id}" \
    "resilience-baseline-async-poll-request" \
    "resilience-baseline-async-trace"
  capture_resilience_trace "${artifact_dir}" "baseline_sync_trace" "resilience-baseline-sync-trace"
  capture_resilience_trace "${artifact_dir}" "baseline_async_trace" "resilience-baseline-async-trace"
  capture_resilience_metrics "${artifact_dir}" "baseline"

  compose_resilience pause server
  sleep 0.5
  capture_request \
    "${artifact_dir}" \
    "api_paused_extract" \
    "POST" \
    "$(resilience_gateway_url)/v1/extract" \
    "${JOINT_PROOF_USER_KEY}" \
    "${payload}" \
    "" \
    "resilience-api-paused-request" \
    "resilience-api-paused-trace"
  capture_resilience_state "${artifact_dir}" "api_paused"
  compose_resilience unpause server
  wait_for_url "$(resilience_gateway_url)/readyz" 180

  compose_resilience stop server
  sleep 1
  capture_request "${artifact_dir}" "api_down_readyz" "GET" "$(resilience_gateway_url)/readyz" ""
  capture_request \
    "${artifact_dir}" \
    "api_down_extract" \
    "POST" \
    "$(resilience_gateway_url)/v1/extract" \
    "${JOINT_PROOF_USER_KEY}" \
    "${payload}" \
    "" \
    "resilience-api-down-request" \
    "resilience-api-down-trace"
  capture_resilience_state "${artifact_dir}" "api_down"
  compose_resilience start server
  wait_for_url "$(resilience_backend_url)/healthz" 180
  wait_for_url "$(resilience_gateway_url)/readyz" 180
  capture_request \
    "${artifact_dir}" \
    "api_recovery_extract" \
    "POST" \
    "$(resilience_gateway_url)/v1/extract" \
    "${JOINT_PROOF_USER_KEY}" \
    "${payload}" \
    "" \
    "resilience-api-recovery-request" \
    "resilience-api-recovery-trace"
  capture_resilience_trace "${artifact_dir}" "api_recovery_trace" "resilience-api-recovery-trace"
  capture_resilience_state "${artifact_dir}" "api_recovery"

  compose_resilience stop joint_worker
  sleep 0.5
  capture_request \
    "${artifact_dir}" \
    "worker_down_async_submit" \
    "POST" \
    "$(resilience_gateway_url)/v1/extract/jobs" \
    "${JOINT_PROOF_USER_KEY}" \
    "${payload}" \
    "" \
    "resilience-worker-down-submit-request" \
    "resilience-worker-recovery-trace"
  local worker_job_id
  worker_job_id="$(json_field "${artifact_dir}/worker_down_async_submit.body.json" "job_id")"
  sleep 0.5
  if [[ -z "${worker_job_id}" ]]; then
    poll_resilience_job_until_terminal \
      "${artifact_dir}" \
      "worker_down_job_status" \
      "${worker_job_id}" \
      "resilience-worker-down-poll-request" \
      "resilience-worker-recovery-trace" \
      1
  else
    capture_request \
      "${artifact_dir}" \
      "worker_down_job_status" \
      "GET" \
      "$(resilience_gateway_url)/v1/extract/jobs/${worker_job_id}" \
      "${JOINT_PROOF_USER_KEY}" \
      "" \
      "" \
      "resilience-worker-down-poll-request" \
      "resilience-worker-recovery-trace"
  fi
  capture_resilience_state "${artifact_dir}" "worker_down"
  compose_resilience start joint_worker
  poll_resilience_job_until_terminal \
    "${artifact_dir}" \
    "worker_recovery_job_status" \
    "${worker_job_id}" \
    "resilience-worker-recovery-poll-request" \
    "resilience-worker-recovery-trace" \
    120
  capture_resilience_trace "${artifact_dir}" "worker_recovery_trace" "resilience-worker-recovery-trace"
  capture_resilience_state "${artifact_dir}" "worker_recovery"

  compose_resilience stop redis
  sleep 1
  capture_request "${artifact_dir}" "redis_down_readyz" "GET" "$(resilience_gateway_url)/readyz" ""
  capture_request \
    "${artifact_dir}" \
    "redis_down_async_submit" \
    "POST" \
    "$(resilience_gateway_url)/v1/extract/jobs" \
    "${JOINT_PROOF_USER_KEY}" \
    "${payload}" \
    "" \
    "resilience-redis-down-submit-request" \
    "resilience-redis-down-trace"
  capture_resilience_state "${artifact_dir}" "redis_down"
  compose_resilience start redis
  wait_resilience_redis
  compose_resilience restart server joint_worker
  wait_for_url "$(resilience_backend_url)/healthz" 180
  wait_for_url "$(resilience_gateway_url)/readyz" 180
  capture_request "${artifact_dir}" "redis_recovery_readyz" "GET" "$(resilience_gateway_url)/readyz" ""
  capture_request \
    "${artifact_dir}" \
    "redis_recovery_extract" \
    "POST" \
    "$(resilience_gateway_url)/v1/extract" \
    "${JOINT_PROOF_USER_KEY}" \
    "${payload}" \
    "" \
    "resilience-redis-recovery-request" \
    "resilience-redis-recovery-trace"
  capture_resilience_state "${artifact_dir}" "redis_recovery"

  compose_resilience stop postgres
  sleep 1
  capture_request "${artifact_dir}" "postgres_down_readyz" "GET" "$(resilience_gateway_url)/readyz" ""
  capture_request \
    "${artifact_dir}" \
    "postgres_down_extract" \
    "POST" \
    "$(resilience_gateway_url)/v1/extract" \
    "${JOINT_PROOF_USER_KEY}" \
    "${payload}" \
    "" \
    "resilience-postgres-down-request" \
    "resilience-postgres-down-trace"
  capture_resilience_state "${artifact_dir}" "postgres_down"
  compose_resilience start postgres
  wait_resilience_postgres
  compose_resilience restart server joint_worker
  wait_for_url "$(resilience_backend_url)/healthz" 180
  wait_for_url "$(resilience_gateway_url)/readyz" 180
  capture_request "${artifact_dir}" "postgres_recovery_readyz" "GET" "$(resilience_gateway_url)/readyz" ""
  capture_request \
    "${artifact_dir}" \
    "postgres_recovery_extract" \
    "POST" \
    "$(resilience_gateway_url)/v1/extract" \
    "${JOINT_PROOF_USER_KEY}" \
    "${payload}" \
    "" \
    "resilience-postgres-recovery-request" \
    "resilience-postgres-recovery-trace"
  capture_resilience_state "${artifact_dir}" "postgres_recovery"

  capture_resilience_metrics "${artifact_dir}" "final"
  capture_resilience_logs "${artifact_dir}"
  write_resilience_manifest "${artifact_dir}"
  clean_artifacts "${artifact_dir}"
  echo "Joint resilience artifacts: ${artifact_dir}"
}

cmd_verify_containerized_llama() {
  need_cmd docker
  need_cmd curl
  ensure_docker_ready
  need_file "${GATEWAY_COMPOSE_FILE}"
  need_file "${JOINT_CONTAINER_LLAMA_ENV_FILE}"
  clear_artifact_dir "${JOINT_CONTAINER_LLAMA_ARTIFACT_DIR}"
  write_containerized_llama_effective_env

  local api_key admin_key
  api_key="$(env_file_value "${JOINT_CONTAINER_LLAMA_EFFECTIVE_ENV_FILE}" "API_KEY")"
  admin_key="$(env_file_value "${JOINT_CONTAINER_LLAMA_EFFECTIVE_ENV_FILE}" "ADMIN_API_KEY")"
  if [[ -z "${api_key}" || -z "${admin_key}" ]]; then
    echo "Containerized llama joint workflow requires API_KEY and ADMIN_API_KEY in ${JOINT_CONTAINER_LLAMA_EFFECTIVE_ENV_FILE}" >&2
    exit 1
  fi

  trap cmd_down_containerized_llama EXIT
  compose_containerized_llama up -d --build --remove-orphans postgres redis llama_server server_llama
  wait_for_url "http://127.0.0.1:${JOINT_CONTAINER_LLAMA_API_PORT}/healthz" 240
  compose_containerized_llama exec -T server_llama python -m alembic -c /app/server/alembic.ini upgrade head
  seed_containerized_llama_keys
  wait_for_url "http://127.0.0.1:${JOINT_CONTAINER_LLAMA_API_PORT}/readyz" 240
  compose_containerized_llama up -d --build --remove-orphans worker_llama gateway_llama
  wait_for_url "http://127.0.0.1:${JOINT_CONTAINER_LLAMA_GATEWAY_PORT}/healthz" 180
  wait_for_url "http://127.0.0.1:${JOINT_CONTAINER_LLAMA_GATEWAY_PORT}/readyz" 180

  env \
    LLM_EXTRACTION_PLATFORM_BASE_URL="http://127.0.0.1:${JOINT_CONTAINER_LLAMA_API_PORT}" \
    LLM_EXTRACTION_PLATFORM_API_KEY="${api_key}" \
    LLM_EXTRACTION_PLATFORM_ADMIN_API_KEY="${admin_key}" \
    GATEWAY_BASE_URL="http://127.0.0.1:${JOINT_CONTAINER_LLAMA_GATEWAY_PORT}" \
    "${ISG_REPO_ROOT}/proof/generate_llm_extraction_platform_observability_pack.sh" \
    "${JOINT_CONTAINER_LLAMA_ARTIFACT_DIR}"
  clean_artifacts "${JOINT_CONTAINER_LLAMA_ARTIFACT_DIR}"

  compose_containerized_llama ps >"${JOINT_CONTAINER_LLAMA_ARTIFACT_DIR}/compose.ps.txt" 2>&1 || true
  compose_containerized_llama logs --no-color --tail=200 server_llama worker_llama llama_server gateway_llama \
    >"${JOINT_CONTAINER_LLAMA_ARTIFACT_DIR}/compose.logs.txt" 2>&1 || true
  python3 - <<'PY' "${JOINT_CONTAINER_LLAMA_ARTIFACT_DIR}"
import json
import sys
from pathlib import Path

artifact_dir = Path(sys.argv[1])
manifest_path = artifact_dir / "manifest.json"
manifest = json.loads(manifest_path.read_text())
manifest["mode"] = "joint_containerized_llama_extract"
manifest["runtime"] = {
    "backend": "containerized llama.cpp",
    "api": "containerized LLMEP API",
    "worker": "containerized LLMEP async worker",
    "gateway": "containerized inference-serving-gateway",
    "acceleration": "cpu",
}
manifest["checks"]["containerized_llama_extract_response_present"] = (
    artifact_dir / "extract.body.json"
).exists()
manifest["checks"]["containerized_llama_async_response_present"] = (
    artifact_dir / "job_status.body.json"
).exists()
manifest["artifacts"]["compose_ps"] = "compose.ps.txt"
manifest["artifacts"]["compose_logs"] = "compose.logs.txt"
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
PY
  echo "Joint containerized live llama artifacts: ${JOINT_CONTAINER_LLAMA_ARTIFACT_DIR}"
}

cmd_verify_kind() {
  need_cmd docker
  need_cmd kind
  need_cmd kubectl
  ensure_docker_ready
  clear_artifact_dir "${JOINT_KIND_ARTIFACT_DIR}"

  cleanup() {
    PHASE2_KIND_WORKFLOW="${JOINT_KIND_WORKFLOW:-live}" \
      PHASE2_KIND_ENV_FILE="${JOINT_KIND_ENV_FILE}" \
      PHASE2_KIND_CLUSTER="${JOINT_KIND_CLUSTER}" \
      "${ISG_REPO_ROOT}/proof/run_kind_stack.sh" down || true
  }
  trap cleanup EXIT

  PHASE2_KIND_WORKFLOW="${JOINT_KIND_WORKFLOW:-live}" \
    PHASE2_KIND_ENV_FILE="${JOINT_KIND_ENV_FILE}" \
    PHASE2_KIND_CLUSTER="${JOINT_KIND_CLUSTER}" \
    "${ISG_REPO_ROOT}/proof/run_kind_stack.sh" up
  PHASE2_KIND_WORKFLOW="${JOINT_KIND_WORKFLOW:-live}" \
    PHASE2_KIND_ENV_FILE="${JOINT_KIND_ENV_FILE}" \
    PHASE2_KIND_CLUSTER="${JOINT_KIND_CLUSTER}" \
    "${ISG_REPO_ROOT}/proof/run_kind_stack.sh" proof

  local source_dir="${ISG_REPO_ROOT}/proof/artifacts/kind_stack"
  if [[ -d "${source_dir}" ]]; then
    cp -R "${source_dir}/." "${JOINT_KIND_ARTIFACT_DIR}/"
  fi
  clean_artifacts "${JOINT_KIND_ARTIFACT_DIR}"
  python3 - <<'PY' "${JOINT_KIND_ARTIFACT_DIR}" "${JOINT_KIND_WORKFLOW:-live}"
import json
import sys
from pathlib import Path

artifact_dir = Path(sys.argv[1])
workflow = sys.argv[2]

def load_json(relative_path: str) -> dict:
    path = artifact_dir / relative_path
    if not path.exists():
        return {}
    try:
      return json.loads(path.read_text())
    except json.JSONDecodeError:
      return {}

sync_extract = load_json("observability_latest/extract.body.json")
async_status = load_json("observability_latest/job_status.body.json")
sync_model = str(sync_extract.get("model", ""))
async_model = str(async_status.get("model", ""))
summary = {
    "mode": f"joint_kind_{workflow}",
    "source": "inference-serving-gateway/proof/run_kind_stack.sh",
    "artifact_root": str(artifact_dir),
    "observability_manifest": "observability_latest/manifest.json",
    "workflow": workflow,
    "checks": {
        "observability_manifest_present": (artifact_dir / "observability_latest" / "manifest.json").exists(),
        "jaeger_services_present": (artifact_dir / "jaeger-services.json").exists(),
        "runtime_workflow_present": (artifact_dir / "runtime" / "workflow.env").exists(),
        "models_status_present": (artifact_dir / "runtime" / "models_status.json").exists(),
        "llama_health_present": (artifact_dir / "runtime" / "llama_health.json").exists(),
        "llama_models_present": (artifact_dir / "runtime" / "llama_models.json").exists(),
        "llama_logs_present": (artifact_dir / "runtime" / "llama-server.logs.txt").exists(),
        "sync_extract_uses_llama_model": "llama.cpp/" in sync_model,
        "async_extract_uses_llama_model": "llama.cpp/" in async_model,
    },
}
(artifact_dir / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
PY
  echo "Joint kind artifacts: ${JOINT_KIND_ARTIFACT_DIR}"
}

run_gateway_kind_stack() {
  PHASE2_KIND_WORKFLOW="${JOINT_KIND_WORKFLOW:-live}" \
    PHASE2_KIND_ENV_FILE="${JOINT_KIND_ENV_FILE}" \
    PHASE2_KIND_CLUSTER="${JOINT_KIND_CLUSTER}" \
    "${ISG_REPO_ROOT}/proof/run_kind_stack.sh" "$@"
}

cmd_kind_up() {
  need_cmd docker
  need_cmd kind
  need_cmd kubectl
  ensure_docker_ready
  run_gateway_kind_stack up
}

cmd_kind_status() {
  need_cmd kubectl
  run_gateway_kind_stack status
}

cmd_kind_smoke() {
  need_cmd kubectl
  need_cmd curl
  run_gateway_kind_stack smoke
}

cmd_kind_down() {
  run_gateway_kind_stack down
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
    verify-observability) shift; cmd_verify_observability "$@" ;;
    verify-edge-controls) shift; cmd_verify_edge_controls "$@" ;;
    verify-llama) shift; cmd_verify_llama "$@" ;;
    verify-containerized) shift; cmd_verify_containerized "$@" ;;
    verify-containerized-llama) shift; cmd_verify_containerized_llama "$@" ;;
    verify-resilience) shift; cmd_verify_resilience "$@" ;;
    kind-up) shift; cmd_kind_up "$@" ;;
    kind-status) shift; cmd_kind_status "$@" ;;
    kind-smoke) shift; cmd_kind_smoke "$@" ;;
    kind-down) shift; cmd_kind_down "$@" ;;
    verify-kind) shift; cmd_verify_kind "$@" ;;
    ""|-h|--help|help) usage ;;
    *)
      echo "Unknown command: ${cmd}" >&2
      usage
      exit 2
      ;;
  esac
}

main "$@"
