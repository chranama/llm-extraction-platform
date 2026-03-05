# 09) CI Hardening

This document defines CI quality expectations for public portfolio credibility.

## Core principles

1. No temporary/noop quality checks in core lanes.
2. Fast reproducibility via local smoke script.
3. Failure artifacts must be informative, not just red/green.
4. Live lanes are optional and clearly marked as conditional evidence.

## Required checks (recommended branch protection)

- `docs (quality)`
- `server (unit)`
- `server (integration)`
- `policy (unit)`
- `policy (integration)`
- `eval (unit)`
- `eval (integration-contract)`
- `ui (tests)`
- `ui (e2e)`
- `integrations (eval-job)`
- `integrations (policy-job)`
- `integrations (e2e-loop)`

## Formatting policy

Core Python lanes should enforce `black --check` and `ruff check`.

If formatting fails:
- run `uv run black <paths>` in affected package,
- rerun package tests,
- rerun CI smoke script.

## Failure artifact bundle

On failure, CI should upload a compact artifact containing:
- runner + version snapshot
- package dependency snapshot
- git status summary
- relevant service logs for integration lanes

This provides interview-reviewable operational traceability.

## Local CI-core reproduction

Use:
```bash
scripts/ci_smoke_matrix.sh
```

The script should run docs checks plus core package tests/lint in a deterministic order.

## Related docs

- [00-testing.md](00-testing.md)
- [05-operations-runbook.md](05-operations-runbook.md)
