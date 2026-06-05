# Testing

The repository uses package-level tests for subsystem behavior and repo-level
integration tests for cross-service workflows.

## Test Layout

- `server/tests/unit/`: API, runtime, application, worker, telemetry, and config units.
- `server/tests/integration/`: API behavior, readiness, auth, policy, extraction, jobs, traces, and reports.
- `policy/tests/`: policy unit and integration coverage.
- `eval/tests/`: evaluation unit and contract/integration coverage.
- `contracts/tests/`: shared contract coverage.
- `integrations/`: cross-service eval, policy, live server, and end-to-end lanes.
- `ui/`: frontend unit, build, and Playwright coverage.
- `cli/tests/`, `config/tests/`, `deploy/tests/`, `schemas/tests/`, `tools/tests/`: auxiliary repo coverage.

## Local Commands

Repo-level auxiliary tests:

```bash
uv run python -m pytest -q cli/tests config/tests contracts/tests schemas/tests tools/tests
```

Server:

```bash
cd server
uv sync --extra test
uv run python -m pytest -q tests/unit
uv run python -m pytest -q tests/integration
```

Policy:

```bash
uv run --project policy --extra test pytest -q
```

Eval:

```bash
uv run --project eval --extra test pytest -q
```

Integrations:

```bash
cd integrations
uv sync --extra test
uv run pytest -q
```

## CI Lanes

`.github/workflows/ci.yml` runs:

- docs quality checks
- server unit and integration tests
- policy unit and integration tests
- eval unit and integration-contract tests
- integration matrix lanes
- UI tests, build, and Playwright e2e
- auxiliary repo tests
- runtime evidence validation

Live eval and live integration workflows are split into separate workflow files
because they require external runtime configuration.

## Behavior Coverage

Important covered behavior includes:

- generation policy enforcement
- extraction capability gates
- schema registry behavior
- sync and async extraction
- durable job state
- readiness and health semantics
- trace inspection
- admin policy reloads
- Prometheus scrape validation
- Grafana dashboard provisioning and populated-query evidence
- direct and proxied local ops-surface checks
- failure and blocked behavior

Tests should stay behavior-named. A reviewer should be able to infer the
important runtime paths from the test filenames before reading the test bodies.
