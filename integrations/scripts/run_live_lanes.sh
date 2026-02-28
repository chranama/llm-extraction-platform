#!/usr/bin/env bash
set -euo pipefail

# Run integrations lanes that require a live server deployment.
# Config via env:
#   INTEGRATIONS_BASE_URL (required)
#   API_KEY               (required)
#   INTEGRATIONS_MODE     (full|generate_only|auto) default full
#   INTEGRATIONS_TIMEOUT  (default 30)
#   PYTEST_ARGS           (optional explicit override)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

need_cmd curl

export INTEGRATIONS_BASE_URL
export API_KEY
export INTEGRATIONS_MODE
export INTEGRATIONS_TIMEOUT

: "${INTEGRATIONS_BASE_URL:?INTEGRATIONS_BASE_URL is required for live lanes}"
: "${API_KEY:?API_KEY is required for live lanes}"
: "${INTEGRATIONS_MODE:=full}"
: "${INTEGRATIONS_TIMEOUT:=30}"

case "${INTEGRATIONS_MODE}" in
  full|generate_only|auto) ;;
  *)
    die "INTEGRATIONS_MODE must be one of: full, generate_only, auto"
    ;;
esac

if [[ -z "${PYTEST_ARGS:-}" ]]; then
  if [[ "${INTEGRATIONS_MODE}" == "full" ]]; then
    PYTEST_ARGS="test_server_live/test_full test_e2e_loop/test_eval_to_policy_loop.py::test_e2e_live_eval_to_policy_extract_only_contract_linkage"
  elif [[ "${INTEGRATIONS_MODE}" == "generate_only" ]]; then
    PYTEST_ARGS="test_server_live/test_generate_only"
  else
    # auto keeps legacy behavior (runs all server_live tests; may be noisy if deployment is mode-specific)
    PYTEST_ARGS="test_server_live"
  fi
fi

print_config
log "Waiting for /healthz..."
"${SCRIPT_DIR}/wait_http.sh" --base-url "${INTEGRATIONS_BASE_URL}" --path "/healthz" --timeout 120

if curl -fsS "${INTEGRATIONS_BASE_URL%/}/readyz" >/dev/null 2>&1; then
  log "Waiting for /readyz..."
  "${SCRIPT_DIR}/wait_http.sh" --base-url "${INTEGRATIONS_BASE_URL}" --path "/readyz" --timeout 180
fi

log "Running live lanes with PYTEST_ARGS=${PYTEST_ARGS}"
run_pytest
