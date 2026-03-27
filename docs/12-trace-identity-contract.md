# 12) Trace Identity Contract

This document summarizes the identity contract used when `llm-extraction-platform` runs behind `inference-serving-gateway`.

Canonical cross-repo reference:

- [Gateway Trace Identity Contract](../../inference-serving-gateway/docs/trace-identity-contract.md)

Use this page as the backend-side quick reference.

## Core Meanings

### `request_id`

- one ID per concrete HTTP request
- poll requests get their own `request_id`
- best for exact request-level access logging

### `trace_id`

- one stable ID for the full logical operation
- in gateway-backed mode, this should come from the trusted inbound `X-Trace-ID`
- async submit, worker, and poll all share the same `trace_id`

### `job_id`

- one stable ID for the async job entity
- only exists for async flows

## Backend Trust Rules

The backend should only trust a distinct inbound `X-Trace-ID` when:

- `EDGE_MODE=behind_gateway`
- `X-Gateway-Proxy: inference-serving-gateway` is present

Otherwise:

- the backend may fall back to backend-local trace behavior

## Surface Semantics In This Repo

### Access logs

- all HTTP requests
- includes async poll requests

### Trace events

- lifecycle timeline
- includes:
  - submit
  - worker execution
  - `extract_job.status_polled`
  - completion/failure

### Inference logs

- actual execution attempts only
- should not be treated as a generic request log surface
- surfaced through `/v1/admin/logs`
- async poll requests are not required to appear here, even when they are visible in:
  - backend access logs
  - backend trace events
  - gateway access logs in integrated runs

## Async Mental Model

One async extraction should look like:

- submit:
  - `request_id = submit-request`
  - `trace_id = shared-trace`
  - `job_id = async-job`

- worker:
  - `trace_id = shared-trace`
  - `job_id = async-job`

- poll:
  - `request_id = poll-request`
  - `trace_id = shared-trace`
  - `job_id = async-job`

That means:

- request identity can vary
- trace identity should remain stable
- job identity remains stable for the async object
