# justfile (v3) — docker compose profiles + env-file matrix
# Requires: docker (compose v2), bash, optional: uv, curl, python
# Install: brew install just

# -----------------------
# Config (overridable)
# -----------------------
ENV_FILE := ".env"
PROJECT_NAME := "llm-extraction-platform"

COMPOSE_DIR := "deploy/compose"
COMPOSE_YML := COMPOSE_DIR + "/docker-compose.yml"
BACKEND_DIR := "backend"
TOOLS_DIR := "tools"
COMPOSE_DOCTOR := TOOLS_DIR + "/compose_doctor.sh"

# Models
MODELS_FULL := "config/models.full.yaml"
MODELS_GENERATE_ONLY := "config/models.generate-only.yaml"

# Host ports (published ports only; DB/Redis are NOT published)
API_PORT := "8000"
UI_PORT := "5173"
PGADMIN_PORT := "5050"
PROM_PORT := "9090"
GRAFANA_PORT := "3000"
PROM_HOST_PORT := "9091"

# DB defaults (for exec-inside-postgres helpers)
PG_USER := "llm"
PG_DB := "llm"

# Optional runtime args
EVAL_ARGS := ""

# Compose command (single file + profiles)
# NOTE: keep as a single string to avoid quoting hell.
compose := "COMPOSE_PROJECT_NAME={{PROJECT_NAME}} docker compose --env-file {{ENV_FILE}} -f {{COMPOSE_YML}}"

# Helper: run compose with profiles
# Usage: just dc "infra api" "up -d --build"
dc profiles args:
  bash -lc "{{compose}} {{#profiles}}{{#each (split profiles \" \")}} --profile {{.}}{{/each}}{{/profiles}} {{args}}"

# -----------------------
# Setup
# -----------------------
default:
  @just --list

init: init-env

init-env:
  bash -lc '\
    if [ -f .env ]; then \
      echo "✔ .env already exists (using {{ENV_FILE}})"; \
    elif [ -f .env.example ]; then \
      echo "📄 Creating .env from .env.example"; \
      cp .env.example .env; \
    else \
      echo "❌ .env.example not found; create .env manually."; \
      exit 1; \
    fi; \
    echo "✅ Env file ready: .env"'

env:
  bash -lc '\
    set -a; [ -f "{{ENV_FILE}}" ] && . "{{ENV_FILE}}"; set +a; \
    echo "ENV_FILE={{ENV_FILE}}"; \
    echo "PROJECT_NAME={{PROJECT_NAME}}"; \
    echo "API_PORT={{API_PORT}} UI_PORT={{UI_PORT}}"; \
    echo "PG_USER={{PG_USER}} PG_DB={{PG_DB}}"; \
    echo "PGADMIN_PORT={{PGADMIN_PORT}} PROM_PORT={{PROM_PORT}} GRAFANA_PORT={{GRAFANA_PORT}} PROM_HOST_PORT={{PROM_HOST_PORT}}"; \
    echo "MODELS_YAML=${MODELS_YAML:-}"; \
  '

config:
  bash -lc '{{compose}} config >/dev/null && echo "✅ compose config OK"'

ps:
  bash -lc '{{compose}} ps'

logs:
  bash -lc '{{compose}} logs -f --tail=200'

down:
  bash -lc '{{compose}} down --remove-orphans'

# -----------------------
# Infra (postgres + redis)
# -----------------------
infra-up:
  just dc "infra" "up -d"
  @echo "✅ infra up (postgres/redis)."

infra-down:
  just dc "infra" "down --remove-orphans"
  @echo "✅ infra down"

infra-ps:
  just dc "infra" "ps"

infra-logs:
  just dc "infra" "logs -f --tail=200"

# -----------------------
# API modes (dockerized)
# -----------------------
api-up:
  just dc "infra api" "up -d --build"
  @echo "✅ api up (docker) @ http://localhost:{{API_PORT}}"
  @echo "ℹ️  Capabilities are controlled by MODELS_YAML."

api-down:
  just dc "infra api" "down --remove-orphans"
  @echo "✅ api down"

api-gpu-up:
  just dc "infra api-gpu" "up -d --build"
  @echo "✅ api_gpu up (docker, nvidia)"

api-gpu-down:
  just dc "infra api-gpu" "down --remove-orphans"
  @echo "✅ api_gpu down"

# (2) Generate-only wrappers (MODELS_YAML gating)
api-up-generate-only:
  bash -lc 'export MODELS_YAML="{{MODELS_GENERATE_ONLY}}"; {{compose}} --profile infra --profile api up -d --build'
  @echo "✅ api up (generate-only) @ http://localhost:{{API_PORT}}"
  @echo "ℹ️  MODELS_YAML={{MODELS_GENERATE_ONLY}}"

api-gpu-up-generate-only:
  bash -lc 'export MODELS_YAML="{{MODELS_GENERATE_ONLY}}"; {{compose}} --profile infra --profile api-gpu up -d --build'
  @echo "✅ api_gpu up (generate-only, nvidia)"
  @echo "ℹ️  MODELS_YAML={{MODELS_GENERATE_ONLY}}"

api-logs:
  bash -lc '{{compose}} logs -f --tail=200'

api-ps:
  bash -lc '{{compose}} ps'

# (3) More granular logs/ps helpers
api-logs-only:
  just dc "api api-gpu" "logs -f --tail=200"

api-ps-only:
  just dc "api api-gpu" "ps"

ui-logs:
  just dc "ui" "logs -f --tail=200"

ui-ps:
  just dc "ui" "ps"

obs-logs:
  just dc "obs obs-host" "logs -f --tail=200"

obs-ps:
  just dc "obs obs-host" "ps"

eval-logs:
  just dc "eval eval-host" "logs -f --tail=200"

eval-ps:
  just dc "eval eval-host" "ps"

# Integration-test profile (ephemeral infra + api_itest)
itest-up:
  just dc "itest" "up -d --build"
  @echo "✅ itest up (postgres_itest/redis_itest/api_itest)"

itest-down:
  just dc "itest" "down --remove-orphans"
  @echo "✅ itest down"

itest-logs:
  just dc "itest" "logs -f --tail=200"

itest-ps:
  just dc "itest" "ps"

# -----------------------
# UI / Admin / Obs
# -----------------------
ui-up:
  just dc "ui" "up -d --build"
  @echo "✅ ui up @ http://localhost:{{UI_PORT}}"

ui-down:
  just dc "ui" "down --remove-orphans"
  @echo "✅ ui down"

admin-up:
  just dc "infra admin" "up -d"
  @echo "✅ pgadmin up @ http://localhost:{{PGADMIN_PORT}}"

admin-down:
  just dc "infra admin" "down --remove-orphans"
  @echo "✅ pgadmin down"

obs-up:
  just dc "obs" "up -d"
  @echo "✅ obs up @ prometheus http://localhost:{{PROM_PORT}} | grafana http://localhost:{{GRAFANA_PORT}}"

obs-down:
  just dc "obs" "down --remove-orphans"
  @echo "✅ obs down"

obs-host-up:
  just dc "obs-host" "up -d"
  @echo "✅ obs-host up @ prometheus http://localhost:{{PROM_HOST_PORT}} | grafana http://localhost:{{GRAFANA_PORT}}"

obs-host-down:
  just dc "obs-host" "down --remove-orphans"
  @echo "✅ obs-host down"

# -----------------------
# Golden paths
# -----------------------
dev-cpu: api-up migrate-docker
  @echo "✅ dev-cpu ready"
  @echo "👉 health:   curl -sS http://localhost:{{API_PORT}}/healthz"
  @echo "👉 ready:    curl -sS http://localhost:{{API_PORT}}/readyz"
  @echo "👉 modelz:   curl -sS http://localhost:{{API_PORT}}/modelz"

dev-gpu: api-gpu-up migrate-docker
  @echo "✅ dev-gpu ready"
  @echo "👉 modelz:   curl -sS http://localhost:{{API_PORT}}/modelz"

dev-cpu-generate-only: api-up-generate-only migrate-docker
  @echo "✅ dev-cpu (generate-only) ready"
  @echo "👉 doctor:   just doctor"

dev-gpu-generate-only: api-gpu-up-generate-only migrate-docker
  @echo "✅ dev-gpu (generate-only) ready"
  @echo "👉 doctor:   just doctor"

# dev-local: infra in docker, API on host
dev-local:
  @echo "✅ dev-local guidance"
  @echo "⚠️  Your compose does NOT publish Postgres to host."
  @echo "   If host API needs Postgres on localhost, publish ports or run host API inside docker."
  @echo "👉 run infra:       just infra-up"
  @echo "👉 run api on host: just api-local"
  @echo "👉 eval host api:   just eval-host-run"

# -----------------------
# DB migrations (docker-only)
# -----------------------
migrate-docker:
  bash -lc '\
    set -euo pipefail; \
    svc=""; \
    if {{compose}} ps --services | grep -qx "api"; then svc="api"; fi; \
    if [ -z "$$svc" ] && {{compose}} ps --services | grep -qx "api_gpu"; then svc="api_gpu"; fi; \
    if [ -z "$$svc" ]; then \
      echo "❌ No running api/api_gpu container found. Run: just api-up or just api-gpu-up"; \
      exit 1; \
    fi; \
    echo "▶ running alembic in $$svc"; \
    {{compose}} exec -T "$$svc" python -m alembic upgrade head; \
    echo "✅ migrations applied (docker)"; \
  '

revision-docker m:
  bash -lc '\
    set -euo pipefail; \
    if [ -z "{{m}}" ]; then echo "Usage: just revision-docker \"your message\""; exit 1; fi; \
    svc=""; \
    if {{compose}} ps --services | grep -qx "api"; then svc="api"; fi; \
    if [ -z "$$svc" ] && {{compose}} ps --services | grep -qx "api_gpu"; then svc="api_gpu"; fi; \
    if [ -z "$$svc" ]; then \
      echo "❌ No running api/api_gpu container found. Run: just api-up or just api-gpu-up"; \
      exit 1; \
    fi; \
    {{compose}} exec -T "$$svc" python -m alembic revision --autogenerate -m "{{m}}"; \
  '

# -----------------------
# Compose doctor
# -----------------------
doctor:
  bash -lc '\
    command -v bash >/dev/null || (echo "❌ bash not found"; exit 1); \
    if [ ! -x "{{COMPOSE_DOCTOR}}" ]; then \
      echo "❌ {{COMPOSE_DOCTOR}} not found or not executable."; \
      echo "   Ensure tools/compose_doctor.sh exists and chmod +x it."; \
      exit 1; \
    fi; \
    # ensure ports/vars are visible to the script even if not in env-file \
    export API_PORT="{{API_PORT}}"; \
    export UI_PORT="{{UI_PORT}}"; \
    export PGADMIN_PORT="{{PGADMIN_PORT}}"; \
    export PROM_PORT="{{PROM_PORT}}"; \
    export GRAFANA_PORT="{{GRAFANA_PORT}}"; \
    export PROM_HOST_PORT="{{PROM_HOST_PORT}}"; \
    "{{COMPOSE_DOCTOR}}"; \
  '

# -----------------------
# (1) Smoke tests
# -----------------------
# Smoke philosophy:
# - "doctor" is the main contract check
# - If API_KEY is present, also hit /v1/generate to prove auth + v1 path works

smoke-cpu:
  just dev-cpu
  just doctor
  bash -lc '\
    set -a; [ -f "{{ENV_FILE}}" ] && . "{{ENV_FILE}}"; set +a; \
    if [ -z "${API_KEY:-}" ]; then \
      echo "ℹ️  API_KEY not set; skipping /v1/generate probe."; \
      exit 0; \
    fi; \
    curl -fsS -X POST "http://localhost:{{API_PORT}}/v1/generate" \
      -H "Content-Type: application/json" \
      -H "X-API-Key: ${API_KEY}" \
      --data "{\"prompt\":\"smoke test\",\"max_new_tokens\":16,\"temperature\":0.2}" >/dev/null; \
    echo "✅ /v1/generate probe OK"; \
  '

smoke-cpu-generate-only:
  just dev-cpu-generate-only
  just doctor

smoke-gpu:
  just dev-gpu
  just doctor

smoke-gpu-generate-only:
  just dev-gpu-generate-only
  just doctor

# -----------------------
# Host API runner (optional)
# -----------------------
api-local:
  bash -lc '\
    set -euo pipefail; \
    set -a; [ -f "{{ENV_FILE}}" ] && . "{{ENV_FILE}}"; set +a; \
    cd "{{BACKEND_DIR}}"; \
    APP_ROOT=.. PORT="{{API_PORT}}" uv run serve; \
  '

# -----------------------
# Tests
# -----------------------
test-unit:
  bash -lc '\
    set -euo pipefail; \
    set -a; [ -f "{{ENV_FILE}}" ] && . "{{ENV_FILE}}"; set +a; \
    cd "{{BACKEND_DIR}}"; \
    PYTHONPATH=src uv run pytest -q -m unit; \
  '

test-integration:
  bash -lc '\
    set -euo pipefail; \
    set -a; [ -f "{{ENV_FILE}}" ] && . "{{ENV_FILE}}"; set +a; \
    cd "{{BACKEND_DIR}}"; \
    PYTHONPATH=src uv run pytest -q -m integration; \
  '

test-all:
  bash -lc '\
    set -euo pipefail; \
    set -a; [ -f "{{ENV_FILE}}" ] && . "{{ENV_FILE}}"; set +a; \
    cd "{{BACKEND_DIR}}"; \
    PYTHONPATH=src uv run pytest -q; \
  '

# -----------------------
# Eval
# -----------------------
eval-host-run:
  just dc "eval-host" "run --rm eval_host sh -lc \"eval {{EVAL_ARGS}}\""

eval-host-shell:
  just dc "eval-host" "run --rm --entrypoint sh eval_host"

eval-run: api-up
  just dc "eval" "run --rm eval sh -lc \"eval {{EVAL_ARGS}}\""

eval-shell: api-up
  just dc "eval" "run --rm --entrypoint sh eval"

# -----------------------
# (6) Model download
# -----------------------
# Usage:
#   just models-download mistralai/Mistral-7B-Instruct-v0.3
# or:
#   MODEL_ID=... HF_TOKEN=... just models-download
models-download model_id="":
  bash -lc '\
    set -euo pipefail; \
    if [ -n "{{model_id}}" ]; then \
      "{{TOOLS_DIR}}/models/download-hf-model.sh" "{{model_id}}"; \
    else \
      "{{TOOLS_DIR}}/models/download-hf-model.sh"; \
    fi; \
  '

# -----------------------
# Cleanup
# -----------------------
clean:
  bash -lc '{{compose}} down --remove-orphans'

clean-volumes:
  bash -lc '\
    read -p "This will DELETE volumes (DB/Redis/Grafana). Are you sure? [y/N] " ans; \
    if [ "$$ans" = "y" ] || [ "$$ans" = "Y" ]; then \
      {{compose}} down -v --remove-orphans; \
    else \
      echo "Aborted."; \
    fi \
  '

# DELAY: "nuke" is sharp; keep as Makefile or a separate script for now