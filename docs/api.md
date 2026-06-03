# API

The runtime service exposes model, schema, generation, extraction, health, admin,
and trace-inspection surfaces.

The FastAPI route definitions live in:

- `server/src/llm_server/api/generate.py`
- `server/src/llm_server/api/extract.py`
- `server/src/llm_server/api/models.py`
- `server/src/llm_server/api/health.py`
- `server/src/llm_server/api/admin.py`

## Public Runtime Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/healthz` | Liveness check. |
| `GET` | `/readyz` | Readiness check with DB, Redis, model, and policy snapshots. |
| `GET` | `/modelz` | Runtime model readiness details. |
| `GET` | `/v1/models` | Model list and deployment capabilities. |
| `GET` | `/v1/models/status` | Minimal model status for non-admin clients. |
| `GET` | `/v1/schemas` | Available extraction schemas. |
| `GET` | `/v1/schemas/{schema_id}` | Full schema detail. |
| `POST` | `/v1/generate` | Text generation. |
| `POST` | `/v1/extract` | Synchronous structured extraction. |
| `POST` | `/v1/extract/jobs` | Submit asynchronous extraction work. |
| `GET` | `/v1/extract/jobs/{job_id}` | Poll asynchronous extraction status. |

## Admin And Inspection Endpoints

Admin endpoints cover usage, logs, model loading, policy reload, runtime reload,
trace detail, replay export, and summary reports. They are defined in
`server/src/llm_server/api/admin.py`.

Representative endpoints:

- `/v1/admin/stats`
- `/v1/admin/logs`
- `/v1/admin/traces/{trace_id}`
- `/v1/admin/policy`
- `/v1/admin/policy/reload`
- `/v1/admin/models/status`
- `/v1/admin/reload`

## Contract Locations

- Request and response models are defined close to each API route.
- Shared artifact contracts live in `contracts/src/llm_contracts/`.
- Extraction schemas live in `schemas/`.
- API behavior is covered by `server/tests/unit/` and `server/tests/integration/`.

## Error Behavior

The server uses structured application errors for invalid schema ids, model
capability mismatches, policy blocks, disabled model modes, and infrastructure
readiness failures.

Relevant areas:

- `server/src/llm_server/core/errors.py`
- `server/tests/integration/test_policy_enforcement_integration.py`
- `server/tests/integration/test_capability_enforcement_endpoints_integration.py`
- `server/tests/integration/test_extract_jobs_integration.py`
