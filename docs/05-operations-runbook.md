# 05) Operations Runbook

Quick diagnostics for common failures in local dev and CI.

## 1. Server not ready

Checks:
```bash
curl -sS http://127.0.0.1:8000/readyz | jq .
curl -sS http://127.0.0.1:8000/modelz | jq .
```

If failing:
- verify `DATABASE_URL` / `REDIS_URL`
- verify `MODELS_YAML` path exists
- verify `MODELS_PROFILE` is present in models YAML

## 2. Extract unexpectedly blocked

Checks:
```bash
curl -sS http://127.0.0.1:8000/v1/models -H "X-API-Key: $API_KEY" | jq .
```

Look for selected model capabilities.

Also check policy overlay:
```bash
echo "POLICY_DECISION_PATH=${POLICY_DECISION_PATH:-}"
```

For offline demo isolation, clear it:
```bash
export POLICY_DECISION_PATH=""
```

## 3. CI failing in server integration

Run locally first:
```bash
cd server
uv sync --extra test
uv run python -m pytest -q tests/integration -k <failing_test_name>
```

Common causes:
- stale import paths after refactor
- profile/model config mismatch
- environment assumptions in tests

## 4. Repo integration lane failures

Run target lane directly:
```bash
cd integrations
uv sync
uv run pytest -q test_eval_job
uv run pytest -q test_policy_job
uv run pytest -q test_e2e_loop
```

## 5. Docs quality failures

```bash
bash scripts/check_docs.sh
```

Fix categories:
- missing required top-level README
- missing archive warning banner
- stale canonical token
- broken markdown link

## 6. Demo flow failures

Use wrapper scripts (with diagnostics output):
- `scripts/demo_extract_gate/run_extract_gate_matrix.sh`
- `scripts/demo_extract_gate/run_host_transformers.sh`
- `scripts/demo_extract_gate/run_docker_llama.sh`

Artifacts and diagnostics land under `traffic_out/<run_tag>/`.

## Related docs

- [02-project-demos.md](02-project-demos.md)
- [03-deployment-modes.md](03-deployment-modes.md)
- [00-testing.md](00-testing.md)
