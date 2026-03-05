#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "[smoke] docs"
bash scripts/check_docs.sh

echo "[smoke] server unit"
(
  cd server
  UV_CACHE_DIR=.uv-cache uv run python -m pytest -q -ra --maxfail=1 --disable-warnings tests/unit
)

echo "[smoke] policy unit"
(
  cd policy
  UV_CACHE_DIR=.uv-cache uv run python -m pytest -q -ra --maxfail=1 --disable-warnings tests/unit
)

echo "[smoke] eval unit"
(
  cd eval
  UV_CACHE_DIR=.uv-cache uv run python -m pytest -q -ra --maxfail=1 --disable-warnings tests/unit
)

echo "[smoke] integrations core lanes"
(
  cd integrations
  UV_CACHE_DIR=.uv-cache uv run pytest -q -ra --maxfail=1 --disable-warnings -m "not server_live" test_eval_job test_policy_job test_e2e_loop
)

echo "[smoke] complete"
