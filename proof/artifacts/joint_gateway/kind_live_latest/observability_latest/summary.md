# Integrated Observability Pack

- Sync request ID: `integration-obs-request-1`
- Shared sync trace ID: `integration-obs-trace-1`
- Async submit request ID: `integration-obs-request-2`
- Shared async trace ID: `integration-obs-trace-2`
- Async poll request ID: `integration-obs-request-3`
- Async job ID: `dbd7e4b1aeda4f98b864a0c7787b416f`

Captured surfaces:

- `gateway.metrics.txt`
- `backend.metrics.txt`
- `sync_trace_detail.json`
- `async_trace_detail.json`
- `sync_otel_trace.json`
- `async_otel_trace.json`
- `sync_logs.json`
- `async_logs.json`
- `async_poll_logs.json`
- `manifest.json`

Surface semantics:

- `sync_logs.json` is an execution-log slice queried by shared sync `trace_id`
- `async_logs.json` is an execution-log slice queried by shared async `trace_id` plus `job_id`
- `async_poll_logs.json` is an observation artifact keyed by poll `request_id` and may be empty because polls are not inference executions

What this pack proves:

- request IDs are preserved across gateway and backend surfaces
- shared trace IDs are preserved from gateway injection through backend response and admin trace inspection
- sync extract execution rows are directly joinable by shared `trace_id`
- async worker execution rows are directly joinable by shared `trace_id` and `job_id`
- async extract is inspectable through submit, worker, and poll trace events
- both gateway and backend metrics can be checked from one run

OpenTelemetry note:

- `sync_otel_trace.json` is a Jaeger export for the sync distributed trace
- `async_otel_trace.json` is a Jaeger export for the async submit plus worker continuation trace
- poll requests remain separate HTTP request traces and are not folded into `async_otel_trace.json`
- application `trace_id` remains the logical-operation ID; OTel `TraceId` is a separate transport-level trace identifier
