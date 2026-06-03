#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CANONICAL_DOCS=(
  README.md
  docs/README.md
  docs/architecture.md
  docs/api.md
  docs/testing.md
  docs/operations.md
  docs/runbook.md
  docs/artifacts.md
  docs/scope.md
  CONTRIBUTING.md
)

TOP_LEVEL_READMES=(
  cli/README.md
  config/README.md
  contracts/README.md
  deploy/README.md
  docs/README.md
  eval/README.md
  integrations/README.md
  policy/README.md
  schemas/README.md
  scripts/README.md
  server/README.md
  simulations/README.md
  tools/README.md
  ui/README.md
)

ARCHIVE_DOCS=(archive/docs/archive/*.md)

has_rg=0
if command -v rg >/dev/null 2>&1; then
  has_rg=1
fi

echo "[docs] checking required README inventory"
for f in "${TOP_LEVEL_READMES[@]}"; do
  [[ -f "$f" ]] || { echo "missing required README: $f" >&2; exit 1; }
done

echo "[docs] checking archive banner"
for f in "${ARCHIVE_DOCS[@]}"; do
  [[ -f "$f" ]] || continue
  [[ "$(basename "$f")" == "README.md" ]] && continue
  if [[ "$has_rg" -eq 1 ]]; then
    ok_banner="$(head -n 2 "$f" | rg -q "Historical snapshot; may not reflect current implementation"; echo $?)"
  else
    ok_banner="$(head -n 2 "$f" | grep -q "Historical snapshot; may not reflect current implementation"; echo $?)"
  fi
  if [[ "$ok_banner" -ne 0 ]]; then
    echo "archive banner missing: $f" >&2
    exit 1
  fi
done

echo "[docs] checking stale tokens in canonical docs"
if [[ "$has_rg" -eq 1 ]]; then
  if rg -n --hidden --glob '!docs/archive/**' --glob 'README.md' --glob 'docs/*.md' --glob 'CONTRIBUTING.md' \
    "ticket_v1|make dev-local|make up|llm-server.git" "${CANONICAL_DOCS[@]}"; then
    echo "stale token detected in canonical docs" >&2
    exit 1
  fi
else
  if grep -nE "ticket_v1|make dev-local|make up|llm-server.git" "${CANONICAL_DOCS[@]}"; then
    echo "stale token detected in canonical docs" >&2
    exit 1
  fi
fi

echo "[docs] checking markdown links"
python3 tools/docs/check_markdown_links.py \
  "${CANONICAL_DOCS[@]}" \
  "${TOP_LEVEL_READMES[@]}"

echo "[docs] checks passed"
