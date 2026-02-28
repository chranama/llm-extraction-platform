# Integrations Test Suite

Repo-level integration tests are split into lanes:

- `test_server_live/`: live server contract checks (`generate_only` and `full`).
- `test_eval_job/`: eval CLI artifact contract checks.
- `test_policy_job/`: policy CLI artifact contract checks.
- `test_e2e_loop/`: cross-package workflow checks (`eval -> policy`, plus live variant).

## Run lanes

From repo root:

```bash
PYTHONPATH="$PWD" uv run --project integrations pytest -q integrations/test_eval_job
PYTHONPATH="$PWD" uv run --project integrations pytest -q integrations/test_policy_job
PYTHONPATH="$PWD" uv run --project integrations pytest -q integrations/test_e2e_loop
```

From `integrations/`:

```bash
PYTHONPATH="$(cd .. && pwd)" uv run pytest -q test_eval_job test_policy_job test_e2e_loop
```

Use `API_KEY` and `INTEGRATIONS_BASE_URL` for live-gated cases.

## Live lanes

Run live lanes (mode-aware):

```bash
INTEGRATIONS_BASE_URL="https://your-host" \
API_KEY="***" \
INTEGRATIONS_MODE="full" \
bash integrations/scripts/run_live_lanes.sh
```

`INTEGRATIONS_MODE=full` runs `test_server_live/test_full` plus live e2e loop checks.
`INTEGRATIONS_MODE=generate_only` runs `test_server_live/test_generate_only`.

CI workflow `.github/workflows/integrations-live.yml` uses secrets:

- `INTEGRATIONS_LIVE_BASE_URL` (required)
- `INTEGRATIONS_LIVE_API_KEY` (required)
- `INTEGRATIONS_LIVE_MODE` (optional; defaults to `full`)
