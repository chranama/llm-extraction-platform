# AWS Deployment Contract

This document defines how `llm-extraction-platform` participates in the bounded AWS deployment slice.

Gateway-side reference:

- `/Users/chranama/career/inference-serving-gateway/docs/aws-deployment-contract.md`

Primary planning reference:

- `/Users/chranama/career/job-search/audit/2026-03-28__phase2-3-aws-deployment-slice-implementation-plan.md`

## Purpose

The backend participates in the AWS slice as:

- the API service
- the async worker image/runtime
- the backend-specific Kubernetes overlay owner

This contract keeps the backend aligned with the gateway-led integrated deployment path.

## Canonical AWS Defaults

Environment name:

- `dev`

Primary region:

- `us-east-1`

AWS-target image architecture:

- `linux/amd64`

Kubernetes namespace:

- `llm`

## Backend Runtime Contract

The backend should preserve the same core semantics it already has locally and in `kind`:

- `request_id`
- `trace_id`
- `job_id`
- `EDGE_MODE=behind_gateway`

The AWS slice should not replace these with AWS-native identifiers.

## Managed Data Contract

The first AWS slice assumes:

- `RDS PostgreSQL` for the primary database
- `ElastiCache Redis` for queue/state support

That is an intentional shift away from in-cluster Postgres/Redis for the cloud path.

## Backend Kubernetes Overlay Contract

Canonical backend AWS overlay path:

- `/Users/chranama/career/llm-extraction-platform/deploy/k8s/overlays/aws-eks/`

This path is the AWS counterpart to:

- `/Users/chranama/career/llm-extraction-platform/deploy/k8s/overlays/local-observability-kind/`

At `2.3.1`, this is a scaffolded target path, not a full deployable overlay yet.

## Cost Guardrails

The backend side of the AWS slice should respect the same bounded design:

- one dev environment
- single-AZ first where practical
- bounded observability footprint
- teardown-friendly deployment
- no always-on cost-heavy additions without a clear reviewer payoff

## What This Slice Does Not Yet Require

The first backend AWS slice does not require:

- multi-AZ database posture
- autoscaling beyond a bounded dev setup
- broad secret-synchronization tooling
- production-grade failover posture

Those can be layered in later if they become necessary.
