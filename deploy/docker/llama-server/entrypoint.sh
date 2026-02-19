#!/usr/bin/env bash
# deploy/docker/llama-server/entrypoint.sh
set -euo pipefail

PORT="${LLAMA_SERVER_PORT:-8080}"
MODEL_FILE="${LLAMA_MODEL_FILE:-}"

if [[ -z "${MODEL_FILE}" ]]; then
  echo "LLAMA_MODEL_FILE is required (e.g. /models/your-model.gguf)" >&2
  exit 2
fi

if [[ ! -f "${MODEL_FILE}" ]]; then
  echo "LLAMA_MODEL_FILE not found: ${MODEL_FILE}" >&2
  echo "Hint: mount your host GGUF directory to /models and set LLAMA_MODEL_FILE=/models/<file>.gguf" >&2
  exit 2
fi

CTX="${LLAMA_CTX_SIZE:-4096}"
BATCH="${LLAMA_BATCH:-256}"
UBATCH="${LLAMA_UBATCH:-}"
N_GPU_LAYERS="${LLAMA_N_GPU_LAYERS:-0}"
SEED="${LLAMA_SEED:-0}"
PARALLEL="${LLAMA_PARALLEL:-1}"

if ! [[ "${PARALLEL}" =~ ^[0-9]+$ ]]; then
  echo "LLAMA_PARALLEL must be an integer; got: ${PARALLEL}" >&2
  exit 2
fi
if [[ "${PARALLEL}" -lt 1 ]]; then
  PARALLEL="1"
fi

THREADS="${LLAMA_THREADS:-}"
if [[ -z "${THREADS}" ]]; then
  THREADS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc 2>/dev/null || echo 4)"
fi

MLOCK="${LLAMA_MLOCK:-0}"
NO_MMAP="${LLAMA_NO_MMAP:-0}"
API_KEY="${LLAMA_SERVER_API_KEY:-}"

# Find the server binary
SERVER_BIN=""
for c in \
  "/app/llama-server" \
  "/usr/local/bin/llama-server" \
  "/usr/bin/llama-server" \
  "/app/server" \
  "/usr/local/bin/server" \
  "/usr/bin/server"
do
  if [[ -x "$c" ]]; then
    SERVER_BIN="$c"
    break
  fi
done

if [[ -z "${SERVER_BIN}" ]]; then
  echo "Could not find llama-server binary in image." >&2
  exit 2
fi

# Probe help to decide which flags exist (avoid restart loops on unknown flags)
HELP="$("${SERVER_BIN}" --help 2>&1 || true)"

_has_flag() {
  local flag="$1"
  echo "${HELP}" | grep -q -- "${flag}"
}

ARGS=()
# host/port flags vary; these are common
if _has_flag "--host"; then
  ARGS+=("--host" "0.0.0.0")
fi
if _has_flag "--port"; then
  ARGS+=("--port" "${PORT}")
fi

# Model flag: prefer -m if present, else --model
if _has_flag " -m " || _has_flag "-m," || _has_flag "-m <"; then
  ARGS+=("-m" "${MODEL_FILE}")
elif _has_flag "--model"; then
  ARGS+=("--model" "${MODEL_FILE}")
else
  # Last resort: try -m (most builds)
  ARGS+=("-m" "${MODEL_FILE}")
fi

# Context
if _has_flag " -c " || _has_flag "-c," || _has_flag "-c <"; then
  ARGS+=("-c" "${CTX}")
elif _has_flag "--ctx-size"; then
  ARGS+=("--ctx-size" "${CTX}")
fi

# Threads
if _has_flag " -t " || _has_flag "-t," || _has_flag "-t <"; then
  ARGS+=("-t" "${THREADS}")
elif _has_flag "--threads"; then
  ARGS+=("--threads" "${THREADS}")
fi

# Batch
if _has_flag " -b " || _has_flag "-b," || _has_flag "-b <"; then
  ARGS+=("-b" "${BATCH}")
elif _has_flag "--batch"; then
  ARGS+=("--batch" "${BATCH}")
fi

# GPU layers
if _has_flag "--n-gpu-layers"; then
  ARGS+=("--n-gpu-layers" "${N_GPU_LAYERS}")
elif _has_flag "--gpu-layers"; then
  ARGS+=("--gpu-layers" "${N_GPU_LAYERS}")
fi

# Parallel (optional)
if _has_flag "--parallel"; then
  ARGS+=("--parallel" "${PARALLEL}")
fi

# ubatch (optional)
if [[ -n "${UBATCH}" ]] && _has_flag "--ubatch"; then
  ARGS+=("--ubatch" "${UBATCH}")
fi

# seed (optional)
if [[ -n "${SEED}" ]] && _has_flag "--seed"; then
  ARGS+=("--seed" "${SEED}")
fi

# memory behavior (optional)
if [[ "${MLOCK}" == "1" || "${MLOCK}" == "true" ]] && _has_flag "--mlock"; then
  ARGS+=("--mlock")
fi
if [[ "${NO_MMAP}" == "1" || "${NO_MMAP}" == "true" ]] && _has_flag "--no-mmap"; then
  ARGS+=("--no-mmap")
fi

# API key (optional; do NOT pass unknown flags)
# Many builds support --api-key; older ones may use --api_key or none at all.
if [[ -n "${API_KEY}" ]]; then
  if _has_flag "--api-key"; then
    ARGS+=("--api-key" "${API_KEY}")
  elif _has_flag "--api_key"; then
    ARGS+=("--api_key" "${API_KEY}")
  fi
fi

echo "Starting llama-server:"
echo "  bin=${SERVER_BIN}"
echo "  port=${PORT}"
echo "  model=${MODEL_FILE}"
echo "  ctx=${CTX} threads=${THREADS} batch=${BATCH} ubatch=${UBATCH:-<unset>} n_gpu_layers=${N_GPU_LAYERS} parallel=${PARALLEL}"
exec "${SERVER_BIN}" "${ARGS[@]}"