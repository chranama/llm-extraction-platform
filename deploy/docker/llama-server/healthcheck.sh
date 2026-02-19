#!/usr/bin/env bash
# deploy/docker/llama-server/healthcheck.sh
set -euo pipefail

PORT="${LLAMA_SERVER_PORT:-8080}"

AUTH=()
if [[ -n "${LLAMA_SERVER_API_KEY:-}" ]]; then
  AUTH=(-H "Authorization: Bearer ${LLAMA_SERVER_API_KEY}")
fi

# 1) Prefer /health if available (fast, no model inference)
if curl -fsS "${AUTH[@]}" "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
  exit 0
fi

# 2) Try /v1/models (often available on OpenAI-compatible servers)
if curl -fsS "${AUTH[@]}" "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
  exit 0
fi

# 3) Last resort: tiny completion request
URL="http://127.0.0.1:${PORT}/v1/completions"
payload='{"prompt":"ping","max_tokens":1,"temperature":0}'

resp="$(curl -fsS "${AUTH[@]}" -H "Content-Type: application/json" -d "${payload}" "${URL}" 2>/dev/null || true)"
if [[ -z "${resp}" ]]; then
  exit 1
fi

echo "${resp}" | grep -q '"choices"' || exit 1
exit 0