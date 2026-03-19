# v3 Rollout Plan

This document records the backend rollout plan and completion state for:

- `llm-extraction-platform v3`

Primary contract sources:

- [`service-boundary.inference-serving-gateway.md`](service-boundary.inference-serving-gateway.md)
- [`../../inference-serving-gateway/docs/v1-rollout-plan.md`](../../inference-serving-gateway/docs/v1-rollout-plan.md)

## 1. Release Goal

`llm-extraction-platform v3` should remain a complete standalone service while becoming explicitly compatible with:

- `inference-serving-gateway v1`

Success means:

1. standalone mode still works as it does today
2. sync extract works cleanly behind the gateway
3. async extract submit and status polling work cleanly behind the gateway
4. `request_id` and `trace_id` can diverge without breaking trace continuity
5. backend-owned extraction semantics stay in this repo

## 2. Recommended Scope

### In scope

- explicit backend mode for gateway-backed operation
- distinct inbound `X-Trace-ID` handling
- separate persisted `trace_id` for async jobs
- worker and polling compatibility with separate request and trace identity
- integrated proof for sync and async gateway-backed flows
- tests for both standalone and gateway-backed behavior

### Out of scope

- requiring the gateway for normal operation
- moving extraction semantics or async ownership into the gateway
- large auth redesign
- moving admin/debugging surfaces into the gateway
- `POST /v1/generate` gateway integration

## 3. Current State Assessment

## Implemented now

- backend route surface matches the gateway contract:
  - `POST /v1/extract`
  - `POST /v1/extract/jobs`
  - `GET /v1/extract/jobs/{job_id}`
  - `GET /healthz`
  - `GET /readyz`
- backend auth remains self-contained and authoritative for `v3`
- backend readiness is stable enough for gateway dependency checks
- request middleware accepts a distinct inbound `X-Trace-ID` in `EDGE_MODE=behind_gateway`
- async jobs persist `trace_id` separately from `request_id`
- submit, worker, and polling paths use the persisted trace identity
- responses emit both `X-Request-ID` and `X-Trace-ID`
- split-identity behavior is covered in unit and integration tests
- a repeatable live gateway-backed proof exists

## Intentionally deferred

- edge authentication / API-key validation migration
- making the gateway mandatory for normal operation
- `POST /v1/generate` gateway integration
- fully automated cross-repo CI for the live proof lane

## 4. Remaining Optional Follow-Up

## A. Cross-repo CI automation

The live gateway-backed proof is repeatable, but it still lives outside the default backend test suite and outside the default gateway `go test` flow.

## B. Edge-auth migration remains intentionally deferred

Backend API-key validation, quota consumption, and local RPM limiting remain authoritative for `v3`.

## C. Live-proof fixtures are intentionally lightweight

The gateway-backed live proof uses deterministic fake-model profiles to validate the contract shape without depending on heavyweight model runtime setup.

## 5. Recommended Design Decisions

## A. Keep backend auth authoritative for `v3`

Recommended decision:

- keep `X-API-Key` validation, local RPM limiting, and quota consumption in the backend for `v3`
- let the gateway forward `X-API-Key`
- defer any true edge-auth migration to a later release

Reason:

- it keeps `v3` focused on compatibility rather than auth redesign
- it matches current backend reality
- it keeps standalone mode intact

## B. Add explicit edge mode

Recommended config shape:

- `EDGE_MODE=standalone|behind_gateway`

Recommended default:

- `standalone`

Behavior in `behind_gateway` mode:

- trust inbound `X-Request-ID` and `X-Trace-ID`
- optionally require `X-Gateway-Proxy: inference-serving-gateway` before trusting them fully

Behavior in `standalone` mode:

- retain current behavior

## C. Persist both `request_id` and `trace_id` for async jobs

Recommended async model:

- `request_id`
  - submission request identity
- `trace_id`
  - stable cross-service trace identity for submit, worker, and poll

This lets:

- submit request ID be one value
- worker request ID be internal or reused as needed
- poll request ID be a different value
- trace identity remain stable across all of them

## 6. Implementation Phases

## Phase 1. Config and request identity foundation

Implement:

- explicit gateway-aware config in settings
- request middleware support for:
  - inbound `X-Request-ID`
  - inbound `X-Trace-ID`
- clear branching between:
  - standalone mode
  - behind-gateway mode

Minimum acceptance:

- standalone mode keeps current behavior
- behind-gateway mode preserves separate request and trace IDs in memory

## Phase 2. Async data model changes

Implement:

- add `trace_id` column to `extract_jobs`
- add migration with backfill:
  - `trace_id = request_id` for existing rows
- update ORM model and any helper types

Minimum acceptance:

- all existing jobs remain valid after migration
- new jobs persist both `request_id` and `trace_id`

## Phase 3. Async submit / worker / poll compatibility

Implement:

- `create_extract_job()` accepts both `request_id` and `trace_id`
- submit endpoint stores both values
- submit response returns persisted `trace_id`
- worker context uses persisted `trace_id`
- worker trace events use persisted `trace_id`
- status polling records:
  - current poll request ID
  - persisted job trace ID
- status response returns persisted `trace_id`

Minimum acceptance:

- submit request ID and poll request ID may differ
- trace continuity remains stable across submit, worker, and poll

## Phase 4. Response and error-header cleanup

Implement:

- consider emitting `X-Trace-ID` alongside `X-Request-ID` on:
  - normal responses
  - error responses

This is not strictly required because the gateway can synthesize trace headers, but it improves consistency and standalone clarity.

Minimum acceptance:

- no response path strips or overwrites trace identity incorrectly

## Phase 5. Test and proof hardening

Implement:

- unit tests for middleware behavior in:
  - standalone mode
  - behind-gateway mode
- integration tests for:
  - sync extract with separate request and trace IDs
  - async submit with separate request and trace IDs
  - status polling with a new request ID and stable trace ID
  - admin trace inspection under split identity
- integrated proof lane shared with gateway contract assumptions

Minimum acceptance:

- gateway-backed behavior is covered by repeatable tests
- standalone behavior remains covered and unchanged

## 7. Concrete Code Areas To Change

Likely files:

- `server/src/llm_server/core/config.py`
- `server/src/llm_server/main.py`
- `server/src/llm_server/api/extract.py`
- `server/src/llm_server/services/extract_jobs.py`
- `server/src/llm_server/db/models.py`
- `server/migrations/`
- optionally:
  - `server/src/llm_server/core/errors.py`
  - `server/src/llm_server/core/logging.py`
  - `server/src/llm_server/services/llm_runtime/inference.py`

Likely test files:

- `server/tests/unit/test_main_unit.py`
- `server/tests/integration/test_extract_jobs_integration.py`
- `server/tests/integration/test_trace_inspection_integration.py`
- new gateway-compat integration tests if needed

## 8. Recommended Exit Criteria

`llm-extraction-platform v3` should be considered complete when:

1. standalone mode still works without the gateway
2. behind-gateway mode accepts and preserves distinct inbound request and trace IDs
3. sync extract preserves trace continuity through the gateway
4. async submit persists both `request_id` and `trace_id`
5. async worker events are recorded under the persisted `trace_id`
6. async status polling can use a different request ID without breaking trace continuity
7. admin trace inspection still returns a coherent trace timeline
8. backend auth still works correctly through the gateway
9. an integrated proof lane demonstrates the supported sync and async paths

Current status:

- all nine criteria are satisfied by the implemented code, targeted tests, and live gateway-backed proof

## 9. Suggested Build Order

Recommended order:

1. config + request middleware
2. async schema migration
3. submit / worker / poll code updates
4. tests for split identity
5. integrated proof lane
6. optional response-header polish

This order keeps the most important compatibility work on the critical path and defers cosmetic cleanup until the contract is already real.
