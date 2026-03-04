# Contributing

Thanks for contributing to this repository.

## Development setup

### Python packages
- `server/`, `policy/`, `eval/`, `contracts/`, `integrations/` use `uv`.
- Run commands with package scoping where applicable, for example:

```bash
uv run --project server python -m pytest -q tests/unit
uv run --project policy python -m pytest -q tests/unit
uv run --project eval python -m pytest -q tests/unit
```

### UI package
- `ui/` uses npm + vitest/playwright.

```bash
cd ui
npm ci
npm run test:coverage
npm run test:e2e
```

## Pull request expectations

Before opening a PR:
- tests pass for changed areas,
- lint/format checks pass for changed areas,
- docs are updated when behavior/configuration/commands change.

Keep PRs focused and reviewable.

## Documentation update rules

When changing these areas, update docs in the same PR:
- `deploy/compose/` or profile wiring changes:
  - `docs/03-deployment-modes.md`
  - `deploy/README.md`
- extraction contract/status-code/shape changes:
  - `docs/01-extraction-contract.md`
  - relevant integration tests
- demo scripts or workflows:
  - `docs/02-project-demos.md`
  - `scripts/README.md`
- test layout/CI lane changes:
  - `docs/00-testing.md`
  - `.github/workflows/ci.yml`

Treat docs as product assets; stale docs are considered defects.

## Branch and commit guidance

- Use short-lived feature branches.
- Use clear commit messages (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`).
- If a change is behaviorally significant, include before/after notes in PR description.

## Security

Do not post secrets, tokens, or private infrastructure details in issues or PRs.
