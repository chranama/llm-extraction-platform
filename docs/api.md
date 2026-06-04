# API

The runtime service exposes model, schema, generation, extraction, health,
admin, and trace-inspection surfaces.

Text generation asks a model to produce new text from a prompt. Structured
extraction asks a model to read source text and return fields that match a
declared JSON schema.

When the service is running locally, FastAPI exposes interactive Swagger docs at
`http://localhost:8000/docs`. The raw OpenAPI document is available at
`http://localhost:8000/openapi.json`.

## Authentication

Runtime API calls use the `X-API-Key` header when API key enforcement is enabled:

```http
X-API-Key: <local-api-key>
```

The runbook explains how to start the local runtime and run smoke requests:

- [Runbook](runbook.md)

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

## Generate Text

`POST /v1/generate` accepts a prompt and optional runtime controls. The service
returns generated text plus policy/runtime metadata.

### Request Body

| Field | Type | Required | Description |
|---|---:|---:|---|
| `prompt` | string | yes | Text prompt sent to the selected model. |
| `model` | string or null | no | Optional model id override. |
| `cache` | boolean | no | Enables request/output cache lookup and write. Defaults to `true`. |
| `max_new_tokens` | integer or null | no | Requested generation length. Policy may clamp it. |
| `temperature` | number or null | no | Sampling temperature. |
| `top_p` | number or null | no | Nucleus sampling value. |
| `top_k` | integer or null | no | Top-k sampling value. |
| `stop` | array of strings or null | no | Stop strings passed to generation. |

### Example Request

```bash
curl -fsS -X POST "http://localhost:8000/v1/generate" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ${API_KEY}" \
  --data '{
    "prompt": "Write a two sentence support reply for a delayed shipment.",
    "model": null,
    "cache": true,
    "max_new_tokens": 64,
    "temperature": 0.2,
    "top_p": 1.0
  }'
```

### Example Response

```json
{
  "model": "local-default",
  "output": "Thanks for your patience. Your shipment is delayed, and we are checking the latest carrier status so we can send you an updated delivery window.",
  "cached": false,
  "requested_max_new_tokens": 64,
  "effective_max_new_tokens": 64,
  "policy_generate_max_new_tokens_cap": 256,
  "clamped": false
}
```

### Response Fields

| Field | Type | Description |
|---|---:|---|
| `model` | string | Resolved model id used for the request. |
| `output` | string | Generated text. |
| `cached` | boolean | Whether the response came from cache. |
| `requested_max_new_tokens` | integer or null | Original requested generation length. |
| `effective_max_new_tokens` | integer or null | Runtime value after policy caps. |
| `policy_generate_max_new_tokens_cap` | integer or null | Active policy cap, if configured. |
| `clamped` | boolean | Whether policy reduced the requested token count. |

## Extract Structured Data

`POST /v1/extract` accepts raw text and a schema id. The service prompts the
model to return JSON, validates the JSON against the selected schema, and can
attempt repair when validation fails.

The example below uses `sroie_receipt_v1`, defined in
`schemas/model_output/sroie_receipt_v1.json`.

### Request Body

| Field | Type | Required | Description |
|---|---:|---:|---|
| `schema_id` | string | yes | Extraction schema id, such as `sroie_receipt_v1`. |
| `text` | string | yes | Raw source text or OCR text to extract from. |
| `model` | string or null | no | Optional model id override. |
| `max_new_tokens` | integer or null | no | Maximum tokens for model output. Defaults to `512`. |
| `temperature` | number or null | no | Sampling temperature. Defaults to `0.0`. |
| `cache` | boolean | no | Enables request/output cache lookup and write. Defaults to `true`. |
| `repair` | boolean | no | Enables JSON/schema repair attempt. Defaults to `true`. |

### Example Request

```bash
curl -fsS -X POST "http://localhost:8000/v1/extract" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ${API_KEY}" \
  --data '{
    "schema_id": "sroie_receipt_v1",
    "text": "ACME STORE\n123 MAIN ST\nDATE: 2024-03-10\nTOTAL: $42.18",
    "model": null,
    "max_new_tokens": 512,
    "temperature": 0.0,
    "cache": true,
    "repair": true
  }'
```

### Example Response

```json
{
  "schema_id": "sroie_receipt_v1",
  "model": "local-default",
  "data": {
    "company": "ACME STORE",
    "address": "123 MAIN ST",
    "date": "2024-03-10",
    "total": "$42.18"
  },
  "cached": false,
  "repair_attempted": false
}
```

### Response Fields

| Field | Type | Description |
|---|---:|---|
| `schema_id` | string | Schema used for validation. |
| `model` | string | Resolved model id used for the request. |
| `data` | object | Extracted fields that passed schema validation. |
| `cached` | boolean | Whether the response came from cache. |
| `repair_attempted` | boolean | Whether the service attempted JSON/schema repair. |

## Submit Async Extraction Work

`POST /v1/extract/jobs` accepts the same request body as `/v1/extract`, but
returns immediately with a queued job. Use the returned `poll_path` to check
status.

### Example Submit Request

```bash
curl -fsS -X POST "http://localhost:8000/v1/extract/jobs" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ${API_KEY}" \
  --data '{
    "schema_id": "sroie_receipt_v1",
    "text": "ACME STORE\n123 MAIN ST\nDATE: 2024-03-10\nTOTAL: $42.18",
    "cache": true,
    "repair": true
  }'
```

### Example Submit Response

```json
{
  "job_id": "6f0f9a12d6a74a2b8b2a5a87cfd44a21",
  "trace_id": "6f0f9a12d6a74a2b8b2a5a87cfd44a21",
  "status": "queued",
  "schema_id": "sroie_receipt_v1",
  "model": "local-default",
  "created_at": "2026-06-04T17:35:12.123456+00:00",
  "poll_path": "/v1/extract/jobs/6f0f9a12d6a74a2b8b2a5a87cfd44a21"
}
```

### Poll Job Status

```bash
curl -fsS "http://localhost:8000/v1/extract/jobs/6f0f9a12d6a74a2b8b2a5a87cfd44a21" \
  -H "X-API-Key: ${API_KEY}"
```

Successful jobs include `result`; failed jobs include `error`.

```json
{
  "job_id": "6f0f9a12d6a74a2b8b2a5a87cfd44a21",
  "trace_id": "6f0f9a12d6a74a2b8b2a5a87cfd44a21",
  "status": "succeeded",
  "schema_id": "sroie_receipt_v1",
  "model": "local-default",
  "created_at": "2026-06-04T17:35:12.123456+00:00",
  "started_at": "2026-06-04T17:35:12.223456+00:00",
  "finished_at": "2026-06-04T17:35:13.123456+00:00",
  "cached": false,
  "repair_attempted": false,
  "result": {
    "company": "ACME STORE",
    "address": "123 MAIN ST",
    "date": "2024-03-10",
    "total": "$42.18"
  },
  "error": null
}
```

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

- Request and response models for generation live in `server/src/llm_server/api/generate.py`.
- Request and response models for extraction live in `server/src/llm_server/api/extract.py`.
- Async extraction job request models live in `server/src/llm_server/services/extract_jobs.py`.
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
