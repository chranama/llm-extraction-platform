#!/usr/bin/env bash
set -euo pipefail

# Run repo-level job lanes that do not require an always-live server:
#   - test_eval_job
#   - test_policy_job
#   - test_e2e_loop

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

export INTEGRATIONS_TIMEOUT
: "${INTEGRATIONS_TIMEOUT:=30}"

print_config
log "Running job lanes (eval_job + policy_job + e2e_loop) ..."

# Default selection can be overridden via PYTEST_ARGS.
: "${PYTEST_ARGS:=test_eval_job test_policy_job test_e2e_loop}"
run_pytest
