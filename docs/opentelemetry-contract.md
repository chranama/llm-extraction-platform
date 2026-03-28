# OpenTelemetry Contract

This document defines the Phase 2.2 tracing contract for `llm-extraction-platform`.

## Scope

At this stage, the backend has the bounded Phase 2.2.4 tracing path:

- SDK dependency baseline is present
- W3C propagation is configured
- OTLP/HTTP exporter configuration is defined
- API startup and shutdown own tracing bootstrap lifecycle
- backend request spans are emitted for the main HTTP flows
- bounded child spans are emitted for the main request/application stages
- async submit persists W3C parent context on the job row for worker continuation
- worker spans resume from that persisted context
- poll requests stay separate request traces and use span links back to the async operation when possible

## Identity Model

The backend keeps two tracing concepts separate:

- OpenTelemetry `TraceId` / `SpanId`
- application `trace_id`

They are not interchangeable.

Application identity remains authoritative for the existing system:

- `request_id` = concrete HTTP request
- `trace_id` = logical operation identity
- `job_id` = async job identity

OpenTelemetry adds a standard distributed-tracing carrier around that model.

The backend should therefore keep:

- existing request/trace/job identifiers in logs, admin endpoints, and proof artifacts
- OTel trace context as transport-level distributed trace state

Backend spans carry the application identifiers as span attributes rather than replacing them.

## Propagation

The backend uses standard W3C propagation:

- `traceparent`
- `tracestate` when present
- baggage propagation

This must work in both modes:

- `EDGE_MODE=standalone`
- `EDGE_MODE=behind_gateway`

For async jobs:

- `ExtractJob.otel_parent_context_json` stores the W3C carrier needed for worker continuation
- worker execution resumes from that carrier
- poll requests remain their own HTTP traces and do not fake parent/child lineage

## Export Protocol

The first bounded rollout uses OTLP over HTTP.

The exporter endpoint should be a full absolute traces URL, for example:

- `http://127.0.0.1:4318/v1/traces`

## Environment Variables

- `OTEL_ENABLED`
- `OTEL_SERVICE_NAME`
- `OTEL_EXPORTER_OTLP_ENDPOINT`

## Runtime Safety Rule

Tracing bootstrap must be safe when disabled.

Current behavior:

- if `OTEL_ENABLED` is false, the backend runs normally without an exporter
- if tracing is enabled but `OTEL_EXPORTER_OTLP_ENDPOINT` is empty, the backend logs a warning and runs without an exporter
- invalid OTLP endpoint configuration is rejected during settings validation

## Current Default

The default backend OTel service name is:

- `llm-extraction-platform`

Later slices will extend the trace path into the async worker.
