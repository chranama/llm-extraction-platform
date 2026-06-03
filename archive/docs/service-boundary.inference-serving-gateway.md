# Service Boundary: `inference-serving-gateway` Integration

## Status

This document describes the supported boundary between:

- `inference-serving-gateway v1`
- `llm-extraction-platform v3`

Current supported state:

- `llm-extraction-platform` still runs as a standalone service.
- `llm-extraction-platform v3` also supports gateway-backed operation through `EDGE_MODE=behind_gateway`.
- `inference-serving-gateway v1` fronts the backend for serving/runtime concerns.
- gateway-backed sync and async flows have been validated with real end-to-end proof.
- backend auth remains authoritative for `v3`; edge-auth migration is explicitly deferred.

## Purpose

Define the contract between:
- `inference-serving-gateway` as the edge/runtime service
- `llm-extraction-platform` as the inference application backend

The goal is to separate:
- serving/runtime concerns at the edge
- inference application concerns in this repo

Internal backend architecture reference:

- [Application Runtime Architecture](application-runtime-architecture.md)

## Design Principles

1. `llm-extraction-platform` must remain a complete standalone service.
2. The gateway may front the service, but the backend must not require it.
3. Extraction logic, async job execution, and trace inspection remain backend-owned.
4. Request admission, timeout budgets, coarse policy, and edge observability move outward to the gateway.
5. Cross-service request and trace continuity must work in both sync and async paths.

## Integration Modes

### Mode A: Standalone backend

```text
Client -> llm-extraction-platform
```

### Mode B: Gateway-backed

```text
Client -> inference-serving-gateway -> llm-extraction-platform
```

Both modes remain supported in the implemented integration.

## Responsibility Split

### Gateway-owned responsibilities in `v1`

- request admission control
- request size limits
- rate limiting
- global and per-route concurrency limits
- coarse route allowlists
- client-facing timeout budgets and cancellation
- upstream routing
- request ID and trace ID origination/propagation
- edge metrics, logs, and readiness aggregation

### Backend-owned responsibilities in `v3`

- API-key validation
- quota consumption and local RPM limiting
- generate / extract semantics
- application orchestration
- run identity and async job lifecycle
- schema handling
- model capability gating
- validation and repair behavior
- runtime prompt/routing/planning
- async job persistence and worker execution
- inference logging
- trace event recording and inspection
- replay/export packaging from traces and execution logs
- admin/debugging surfaces

### Deferred beyond the current split release

- edge authentication / API-key validation
- broader model allowlists at the gateway layer

## Supported Backend Routes

The gateway fronts these backend routes:

- `POST /v1/extract`
- `POST /v1/extract/jobs`
- `GET /v1/extract/jobs/{job_id}`
- `GET /healthz`
- `GET /readyz`

Optional later route:
- `POST /v1/generate`

## Header Contract

### Accepted inbound request headers at the gateway

The gateway accepts and normalizes:
- `X-Request-ID`
- `X-Trace-ID`

### Required gateway-to-backend headers

When the gateway proxies a request, it sends:
- `X-Request-ID`
- `X-Trace-ID`
- `X-Gateway-Proxy: inference-serving-gateway`

Optional forwarded headers:
- `X-Forwarded-For`
- `X-Forwarded-Proto`

## Identity Rules

- If the client provides `X-Request-ID`, the gateway preserves it.
- If the client does not provide `X-Request-ID`, the gateway generates it.
- The gateway always ensures a single canonical request ID is sent upstream.
- Trace identity must remain stable across gateway and backend.
- In gateway-backed mode, the backend treats gateway-provided request and trace headers as authoritative when configured to do so.

## Trust Model

`llm-extraction-platform v3` introduces:

- `EDGE_MODE=behind_gateway`

When enabled:
- request and trace headers from the gateway are trusted
- the backend requires `X-Gateway-Proxy: inference-serving-gateway` before trusting a distinct inbound `X-Trace-ID`

When disabled:
- the backend behaves as a standalone service

Even in gateway-backed mode, the backend should retain defensive domain validation.

Canonical identity-semantics reference:

- `12-trace-identity-contract.md`

## Response Contract

The gateway preserves backend:
- status codes
- JSON payloads
- async job IDs
- trace IDs

The gateway emits canonical response headers:
- `X-Request-ID`
- `X-Trace-ID`

## Error Boundary

### Gateway-owned error classes in `v1`

- invalid request
- unsupported route
- request too large
- rate limited
- concurrency limited
- upstream timeout
- upstream unavailable

### Backend-owned error classes

- invalid schema
- extraction failure
- model capability rejection
- validation/repair failure
- async job lifecycle errors

## Timeout and Cancellation Contract

- The gateway owns client-facing timeout budgets.
- The backend may still use internal timeouts for its own protection.
- If the gateway deadline expires, it should cancel the upstream request.
- The backend should tolerate cancellation cleanly.

## Health and Readiness Contract

The backend already exposes:
- `GET /healthz`
- `GET /readyz`

Gateway behavior:
- gateway `healthz` reports gateway process health
- gateway `readyz` depends on both gateway readiness and backend readiness

The backend contract for `readyz` remains stable so the gateway can depend on it.

## Async Contract

For `POST /v1/extract/jobs`:
- the gateway forwards the request with gateway-owned headers added
- the backend returns `job_id` and `trace_id`
- the gateway preserves both
- subsequent `GET /v1/extract/jobs/{job_id}` polls through the gateway must preserve trace continuity

## Observability Contract

### Gateway emits

- edge request logs
- edge latency and rejection metrics
- upstream latency and failure metrics

### Backend emits

- backend access logs for HTTP-request visibility
- inference logs
- request trace events
- admin log surfaces

### Shared correlation keys

- `request_id`
- `trace_id`
- `job_id` for async flows

The design prefers linked edge and backend telemetry rather than moving all request inspection into the gateway.

Interpretation:

- `request_id` identifies one concrete HTTP request
- `trace_id` identifies the full logical operation
- `job_id` identifies the async job entity
- poll requests should be visible in access logs and trace events
- `/v1/admin/logs` should remain execution-focused rather than acting as a catch-all request log surface

## Security and Logging Rules

By default, neither layer should log:
- raw prompt text
- full extracted payloads
- auth tokens or secrets

Structured metadata-only logging is preferred.

## Compatibility State for `llm-extraction-platform v3`

The current integration is considered successful when:

1. `llm-extraction-platform` still runs cleanly as a standalone service
2. it accepts gateway-propagated request and trace headers
3. sync extract requests work through the gateway
4. async extract submit and status polling work through the gateway
5. cross-service trace continuity is preserved
6. no core extraction semantics are moved out of this repo

## Implementation Note

The gateway was developed in isolation and validated first against a mock backend. `llm-extraction-platform v3` then added the gateway-aware changes needed to support both standalone and gateway-backed operation without moving extraction semantics out of this repository.

The backend side is now also organized more explicitly around:

- transport in `api/`
- orchestration in `application/`
- run/domain state in `domain/`
- runtime planning in `runtime/`
- replay/export packaging in `observability/`

That internal refactor does not change the external gateway/backend contract, but it makes the backend side of the split easier to reason about.
