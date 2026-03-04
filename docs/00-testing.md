# Testing and CI

This repository uses a layered test strategy:
- package-level unit tests for `server/`, `policy/`, `eval/`, and `ui/`
- package-level integration tests for service boundaries
- repo-level integration suites under `integrations/`
- optional live suites for externally reachable deployments

## Test layout

### Server
- Unit: `server/tests/unit/`
- Integration: `server/tests/integration/`

### Policy
- Unit: `policy/tests/unit/`
- Integration: `policy/tests/integration/`

### Eval
- Unit: `eval/tests/unit/`
- Integration: `eval/tests/integration/`

### UI
- Unit/component: `ui/src/**/__tests__/`
- Browser E2E: `ui/e2e/`

### Repo-level integrations
- `integrations/test_eval_job/`
- `integrations/test_policy_job/`
- `integrations/test_e2e_loop/`
- `integrations/test_server_live/` (live/optional lane)

## Canonical local commands

### Server
```bash
cd server
uv sync --extra test
uv run python -m pytest -q tests/unit
uv run python -m pytest -q tests/integration
```

### Policy
```bash
cd policy
uv sync --extra test
uv run python -m pytest -q tests/unit
uv run python -m pytest -q tests/integration
```

### Eval
```bash
cd eval
uv sync --extra test
uv run python -m pytest -q tests/unit
uv run python -m pytest -q tests/integration
```

### UI
```bash
cd ui
npm ci
npm run test:coverage
npm run test:e2e
```

### Repo integrations
```bash
cd integrations
uv sync
uv run pytest -q -m "not server_live" test_eval_job test_policy_job test_e2e_loop
```

## CI structure

Primary CI workflow: [`.github/workflows/ci.yml`](../.github/workflows/ci.yml)

Key lanes:
- `server (unit)` and `server (integration)`
- `policy (unit)` and `policy (integration)`
- `eval (unit)` and `eval (integration-contract)`
- `ui (tests)` and `ui (e2e)`
- `integrations (eval-job/policy-job/e2e-loop)`
- `integrations (server-live)`
- `docs (quality)`

Additional live jobs are defined in:
- [`.github/workflows/eval-live.yml`](../.github/workflows/eval-live.yml)
- [`.github/workflows/integrations-live.yml`](../.github/workflows/integrations-live.yml)

## Coverage and quality expectations

Coverage thresholds are enforced by package CI jobs (for example, server/eval/policy unit lanes). Integration suites are assessed by behavioral completeness rather than raw line coverage.

For integration completeness, prefer these checks:
- contract coverage (status codes, payload semantics)
- configuration/deployment profile coverage
- failure-path and recovery-path assertions
- artifact and control-plane wiring (`MODELS_YAML`, policy decisions, SLO inputs)

## Troubleshooting quick map

- If server integration fails on startup, inspect model config/profile wiring (`MODELS_YAML`, `MODELS_PROFILE`) first.
- If policy/eval integration fails, verify contract and schema roots are repo-level (`contracts/` and `schemas/`).
- If UI E2E fails in CI, inspect Playwright artifact upload from the failed lane.
- If repo integration lanes fail, run the specific lane locally before broad reruns.

## Related docs

- [Extraction Contract](01-extraction-contract.md)
- [Project Demos](02-project-demos.md)
- [Deployment Modes](03-deployment-modes.md)
