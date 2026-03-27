# Replay Export Seam

This document defines the first bridge from backend observability to future eval/runtime-comparison work.

## Purpose

The backend can now package a single trace or a single execution log into a minimal replay-oriented manifest.

That does **not** mean every exported case is fully replay-ready yet.

It does mean the system now has a canonical place where replay packaging originates:

- [observability/replay_cases.py](../server/src/llm_server/observability/replay_cases.py)
- [observability/regression_manifests.py](../server/src/llm_server/observability/regression_manifests.py)

## Export surfaces

Admin-only endpoints:

- `GET /v1/admin/replay-cases/traces/{trace_id}`
- `GET /v1/admin/replay-cases/logs/{log_id}`

These return a `regression_replay_manifest_v1` payload with:

- a `source` section
- one `cases` item
- request metadata
- expectation metadata
- correlation fields
- observability metadata

## Replay readiness

Each exported case includes:

- `replay_ready`
- `missing_fields`

This is intentional.

Some traces, especially failures that happen before an execution log is written, do not yet contain enough request payload to be replayed directly. In those cases:

- the export still exists
- the expectation and correlation surfaces are preserved
- the manifest makes the missing replay inputs explicit

## Current role in the architecture

This seam is not the full eval modernization plan.

Its current role is narrower:

- make replay packaging real
- keep it backend-owned
- give future eval/runtime-comparison work a stable export source
