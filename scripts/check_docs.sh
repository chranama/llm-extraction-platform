#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CANONICAL_DOCS=(
  README.md
  PORTFOLIO_ONE_PAGER.md
  docs/README.md
  docs/00-testing.md
  docs/01-extraction-contract.md
  docs/02-project-demos.md
  docs/03-deployment-modes.md
  docs/09-ci-hardening.md
  docs/10-model-decision-memo.md
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

ARCHIVE_DOCS=(docs/archive/*.md)

echo "[docs] checking required README inventory"
for f in "${TOP_LEVEL_READMES[@]}"; do
  [[ -f "$f" ]] || { echo "missing required README: $f" >&2; exit 1; }
done

echo "[docs] checking archive banner"
for f in "${ARCHIVE_DOCS[@]}"; do
  [[ -f "$f" ]] || continue
  [[ "$(basename "$f")" == "README.md" ]] && continue
  if ! head -n 2 "$f" | rg -q "Historical snapshot; may not reflect current implementation"; then
    echo "archive banner missing: $f" >&2
    exit 1
  fi
done

echo "[docs] checking stale tokens in canonical docs"
if rg -n --hidden --glob '!docs/archive/**' --glob 'README.md' --glob 'docs/*.md' --glob 'CONTRIBUTING.md' \
  "ticket_v1|make dev-local|make up|llm-server.git" "${CANONICAL_DOCS[@]}"; then
  echo "stale token detected in canonical docs" >&2
  exit 1
fi

echo "[docs] checking markdown links"
python3 tools/docs/check_markdown_links.py \
  "${CANONICAL_DOCS[@]}" \
  "${TOP_LEVEL_READMES[@]}"

echo "[docs] checks passed"
