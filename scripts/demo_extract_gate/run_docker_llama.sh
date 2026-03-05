#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../" && pwd)"
cd "$ROOT"

API_PORT="${API_PORT:-18020}" \
DOCKER_PROFILE="${DOCKER_PROFILE:-docker-llama}" \
DOCKER_DEMO_MODEL_ID="${DOCKER_DEMO_MODEL_ID:-llama.cpp/SmolLM2-360M-Instruct-Q8_0-GGUF}" \
./scripts/demo_extract_gate/run_extract_gate_matrix.sh --skip-host "$@"
