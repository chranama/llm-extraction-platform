# Service Boundary: Future `inference-serving-gateway` Integration

## Status

This document describes a **future improvement** planned for `llm-extraction-platform` rather than a capability implemented in the current repository state.

Current state:
- `llm-extraction-platform` is a standalone inference application service.
- Clients call the service directly.
- Async jobs, trace inspection, policy enforcement, and extraction semantics are all owned inside this repo.

Planned future state:
- a separate Go service, `inference-serving-gateway`, fronts this backend for serving/runtime concerns
- `llm-extraction-platform` remains valid as a standalone service
- `llm-extraction-platform v3.0.0` becomes explicitly gateway-aware without becoming gateway-dependent

This document is intended to lock the boundary early so the future Go service can be built in isolation.

## Purpose

Define the contract between:
- `inference-serving-gateway` as the future edge/runtime service
- `llm-extraction-platform` as the inference application backend

The goal is to separate:
- serving/runtime concerns at the edge
- inference application concerns in this repo

## Design Principles

1. `llm-extraction-platform` must remain a complete standalone service.
2. The future gateway may front the service, but the backend must not require it.
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

Both modes must remain supported after the future integration work lands.

## Responsibility Split

### Future gateway-owned responsibilities

- request admission control
- request size limits
- edge authentication / API-key validation
- rate limiting
- global and per-route concurrency limits
- coarse model/route allowlists
- client-facing timeout budgets and cancellation
- upstream routing
- request ID and trace ID origination/propagation
- edge metrics, logs, and readiness aggregation

### Backend-owned responsibilities

- generate / extract semantics
- schema handling
- model capability gating
- validation and repair behavior
- async job persistence and worker execution
- inference logging
- trace event recording and inspection
- admin/debugging surfaces

## Supported Backend Routes

The future gateway is expected to front these existing backend routes:

- `POST /v1/extract`
- `POST /v1/extract/jobs`
- `GET /v1/extract/jobs/{job_id}`
- `GET /healthz`
- `GET /readyz`

Optional future route:
- `POST /v1/generate`

## Header Contract

### Accepted inbound request headers

The future gateway may accept and normalize:
- `X-Request-ID`
- `X-Trace-ID`

### Required gateway-to-backend headers

When the future gateway proxies a request, it should send:
- `X-Request-ID`
- `X-Trace-ID`
- `X-Gateway-Proxy: inference-serving-gateway`

Optional forwarded headers:
- `X-Forwarded-For`
- `X-Forwarded-Proto`

## Identity Rules

- If the client provides `X-Request-ID`, the future gateway should preserve it.
- If the client does not provide `X-Request-ID`, the future gateway should generate it.
- The future gateway must always ensure a single canonical request ID is sent upstream.
- Trace identity must remain stable across gateway and backend.
- In future gateway-backed mode, the backend should treat gateway-provided request and trace headers as authoritative when configured to do so.

## Future Trust Model

`llm-extraction-platform v3.0.0` should introduce a gateway-aware config mode such as:

- `TRUST_GATEWAY_HEADERS=true`

or:

- `EDGE_MODE=behind_gateway`

When enabled:
- request and trace headers from the gateway are trusted
- the backend may relax duplicated edge concerns

When disabled:
- the backend behaves as a standalone service

Even in gateway-backed mode, the backend should retain defensive domain validation.

## Response Contract

The future gateway should preserve backend:
- status codes
- JSON payloads
- async job IDs
- trace IDs

The future gateway may add response headers:
- `X-Request-ID`
- `X-Trace-ID`

## Error Boundary

### Future gateway-owned error classes

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

- The future gateway owns client-facing timeout budgets.
- The backend may still use internal timeouts for its own protection.
- If the gateway deadline expires, it should cancel the upstream request.
- The backend should tolerate cancellation cleanly.

## Health and Readiness Contract

The backend already exposes:
- `GET /healthz`
- `GET /readyz`

Future behavior:
- gateway `healthz` reports gateway process health
- gateway `readyz` depends on both gateway readiness and backend readiness

The backend contract for `readyz` should remain stable so the future gateway can depend on it.

## Async Contract

For `POST /v1/extract/jobs`:
- the future gateway forwards the request with gateway-owned headers added
- the backend returns `job_id` and `trace_id`
- the gateway preserves both
- subsequent `GET /v1/extract/jobs/{job_id}` polls through the gateway must preserve trace continuity

## Observability Contract

### Future gateway emits

- edge request logs
- edge latency and rejection metrics
- upstream latency and failure metrics

### Backend emits

- inference logs
- request trace events
- admin log surfaces

### Shared correlation keys

- `request_id`
- `trace_id`

The future design should prefer linked edge and backend telemetry rather than moving all request inspection into the gateway.

## Security and Logging Rules

By default, neither layer should log:
- raw prompt text
- full extracted payloads
- auth tokens or secrets

Structured metadata-only logging is preferred.

## Compatibility Goal for `llm-extraction-platform v3.0.0`

The future integration should be considered successful when:

1. `llm-extraction-platform` still runs cleanly as a standalone service
2. it accepts gateway-propagated request and trace headers
3. sync extract requests work through the gateway
4. async extract submit and status polling work through the gateway
5. cross-service trace continuity is preserved
6. no core extraction semantics are moved out of this repo

## Implementation Note

The future Go gateway should be developed in its own repository and validated first against a mock backend. After that, `llm-extraction-platform v3.0.0` can add the small set of gateway-aware changes needed to support both standalone and gateway-backed operation.
