#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_INTEGRATION="${RUN_INTEGRATION:-0}"

echo "[smoke] docs"
bash scripts/check_docs.sh

echo "[smoke] server lint/format/unit"
(
  cd server
  UV_CACHE_DIR=.uv-cache uv run ruff check .
  UV_CACHE_DIR=.uv-cache uv run black --check src tests
  UV_CACHE_DIR=.uv-cache uv run python -m pytest -q -ra --maxfail=1 --disable-warnings tests/unit
)

echo "[smoke] policy lint/format/unit"
(
  cd policy
  UV_CACHE_DIR=.uv-cache uv run ruff check .
  UV_CACHE_DIR=.uv-cache uv run black --check src tests
  UV_CACHE_DIR=.uv-cache uv run python -m pytest -q -ra --maxfail=1 --disable-warnings tests/unit
)

echo "[smoke] eval lint/format/unit"
(
  cd eval
  UV_CACHE_DIR=.uv-cache uv run ruff check .
  UV_CACHE_DIR=.uv-cache uv run black --check src tests
  UV_CACHE_DIR=.uv-cache uv run python -m pytest -q -ra --maxfail=1 --disable-warnings tests/unit
)

echo "[smoke] integrations core lanes"
(
  cd integrations
  UV_CACHE_DIR=.uv-cache uv run pytest -q -ra --maxfail=1 --disable-warnings -m "not server_live" test_eval_job test_policy_job test_e2e_loop
)

if [[ "$RUN_INTEGRATION" == "1" ]]; then
  echo "[smoke] server integration"
  (
    cd server
    UV_CACHE_DIR=.uv-cache uv run python -m pytest -q -ra --maxfail=1 --disable-warnings tests/integration
  )

  echo "[smoke] policy integration"
  (
    cd policy
    UV_CACHE_DIR=.uv-cache uv run python -m pytest -q -ra --maxfail=1 --disable-warnings tests/integration
  )

  echo "[smoke] eval integration-contract"
  (
    cd eval
    UV_CACHE_DIR=.uv-cache uv run python -m pytest -q -ra --maxfail=1 --disable-warnings -m integration_contract tests/integration
  )
fi

echo "[smoke] complete"
