#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../" && pwd)"
cd "$ROOT"

API_PORT="${API_PORT:-18010}" \
HOST_PROFILE="${HOST_PROFILE:-host-transformers}" \
HOST_DEMO_MODEL_ID="${HOST_DEMO_MODEL_ID:-sshleifer/tiny-gpt2}" \
./scripts/demo_extract_gate/run_phase41.sh --skip-docker "$@"
