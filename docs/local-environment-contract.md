# 13) Phase 2 Local Environment Contract

This note defines the backend-side contract that the integrated local environment should use in Phase 2.

It exists to make the proof stack reproducible without depending on temporary files or remembered shell state.

## Canonical Runtime Contract

The backend and worker should be started with:

```bash
APP_ROOT=/Users/chranama/career/llm-extraction-platform
APP_PROFILE=test
MODELS_PROFILE=observability-proof
MODELS_YAML=/Users/chranama/career/llm-extraction-platform/proof/fixtures/models.observability-proof.yaml
SCHEMAS_DIR=/Users/chranama/career/llm-extraction-platform/schemas/model_output
DATABASE_URL=postgresql+asyncpg://llm:llm@127.0.0.1:5433/llm
REDIS_ENABLED=1
REDIS_URL=redis://127.0.0.1:6379/0
EDGE_MODE=behind_gateway
```

This is the canonical local contract for:

- sync extract proof
- async worker proof
- gateway-integrated observability proof

## Why This Contract Exists

These environment values close the seams uncovered during the live Phase 1 and 1.5 runs:

- `MODELS_YAML` is explicit, so the proof does not depend on fallback model config resolution
- `SCHEMAS_DIR` is explicit, so schema lookup is stable from the server runtime
- `EDGE_MODE=behind_gateway` preserves the shared trace contract
- the proof fixture is checked into the repo instead of being generated in `/tmp`

## Canonical Proof Fixture

Use:

- `/Users/chranama/career/llm-extraction-platform/proof/fixtures/models.observability-proof.yaml`

That fixture:

- uses the fake backend
- allows both generate and extract paths
- emits a schema-valid receipt object for `sroie_receipt_v1`

Expected fake output:

```json
{"company":"ACME","date":"2026-03-25","total":"10.00"}
```

## Schema Contract

Canonical schema directory:

- `/Users/chranama/career/llm-extraction-platform/schemas/model_output`

Canonical observability-proof schema:

- `sroie_receipt_v1`

This matches the gateway proof pack defaults and removes the need for ad hoc payload/schema overrides during local proof runs.

## Identity Contract

This local-environment contract assumes the hardened Phase 1.5 identity model:

- `request_id` = concrete HTTP request
- `trace_id` = full logical operation
- `job_id` = async job entity

Reference:

- `/Users/chranama/career/llm-extraction-platform/docs/12-trace-identity-contract.md`

## Phase 2 Expectation

Phase 2 should treat this file as the backend-side startup contract for the integrated local stack.

That means future local automation should:

- launch the backend and worker with this env by default
- avoid temporary proof models files
- preserve schema-valid extract behavior for the observability pack
