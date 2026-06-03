# AWS Deployment Contract

This document defines how `llm-extraction-platform` participates in the bounded AWS deployment slice.

Gateway-side reference:

- `/Users/chranama/career/inference-serving-gateway/docs/aws-deployment-contract.md`

Primary planning reference:

- `/Users/chranama/career/job-search/audit/2026-03-28__phase2-3-aws-deployment-slice-implementation-plan.md`

Phase 2.2.9 backend runtime reference:

- [Runtime Quality Scorecard](runtime-quality-scorecard.md)

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

Bounded cloud log path:

- backend API and worker logs should flow into the same bounded cloud log surface used by the integrated AWS slice
- `CloudWatch Logs` is the default first-slice target
- correlation fields should remain:
  - `request_id`
  - `trace_id`
  - `job_id`

## Backend Runtime Contract

The backend should preserve the same core semantics it already has locally and in `kind`:

- `request_id`
- `trace_id`
- `job_id`
- `EDGE_MODE=behind_gateway`

The AWS slice should not replace these with AWS-native identifiers.

The backend side of the AWS slice should remain inspectable through one canonical workflow:

- `smoke`
  - one sync request through the gateway-backed `ALB` path
- `inspect`
  - one correlated backend or worker run across logs, metrics, and traces
- `rollback or teardown`
  - one explicit operator path if the deployed behavior is unhealthy

At `2.3.1`, this is a contract-level expectation, not a full runbook yet.

The backend should also carry forward the Phase 2.2.9 runtime-quality contract by making visible:

- the seeded sync and async quality targets that later cloud rollout will reuse
- the current usage-scope seed for metering and quota reasoning
- the quality signals that are allowed to influence runtime choice

## 2.3.1 Backend Quality And Runtime-Choice Contract

The backend is where the first cloud slice proves that the "AI" side of the system is operationally legible.

The backend-side AWS contract should therefore make these runtime-quality signals explicit:

- `extract_contract_pass_rate`
- `structured_output_invalid_rate`
- `repair_attempt_rate`
- `repair_success_rate`
- `policy_rejection_rate`
- `async_completion_before_timeout_rate`
- `rough_request_cost_usd`

Current contract rule:

- a prompt, provider, repair, route, or policy change is not a win unless it preserves or improves contract pass rate while also improving latency, cost, or policy clarity

Current cloud-slice quality thresholds reused from `Phase 2.2.9`:

- contract-valid extract rate:
  - `>= 98%` over the `1h` rollout window
- async completion-before-timeout rate:
  - `>= 95%` over the `1h` rollout window
- sync extract p95 latency:
  - `<= 2.0s` over the `5m` operator window on the bounded proof profile

These do not require new backend metrics to exist on day one.
They do require the backend-side AWS docs and proof path to make the signal sources explicit.

## 2.3.1 Backend Usage, Quota, And Cost Contract

The backend remains authoritative for the first bounded usage scope.

Current first-slice usage scope:

- `api_key`

Current first-slice usage and metering surfaces:

- `GET /v1/me/usage`
- `GET /v1/admin/usage`
- inference logs with token and latency data
- API-key quota state
- proof artifacts and runbook notes

Current first-slice quota or admission-control expectation:

- one proof key or bounded usage scope must have an explicit finite quota, admission rule, or other reviewer-visible fairness control before the AWS slice is considered believable
- this should reuse current backend quota or admission surfaces rather than introduce billing machinery

Current first-slice cost-attribution expectation:

- rough request cost should be derivable from token counts, provider metadata, and usage reports
- AWS billing services are not the primary source of truth for request-level cost reasoning in this slice

Important metric rule:

- raw `api_key` values should stay out of default Prometheus labels even though they are valid in DB-backed usage and admin surfaces

## 2.3.1 Backend Proof And Rollback Evidence

The backend side of the AWS slice should make these proof artifacts possible:

- one sync extract that produces valid structured output
- one async job that completes within the bounded timeout budget
- one correlated trace or log trail showing request, trace, and job identity continuity
- one quality or repair note explaining whether the run passed cleanly, required repair, or was blocked by policy
- one usage or rough-cost snapshot tied to the same bounded usage scope

Rollback or teardown should be justified when:

- the canonical sync proof stops producing contract-valid output
- the canonical async proof stops completing acceptably
- the quality thresholds above are broken after the current change

## Managed Data Contract

The first AWS slice assumes:

- `RDS PostgreSQL` for the primary database
- `ElastiCache Redis` for queue/state support

That is an intentional shift away from in-cluster Postgres/Redis for the cloud path.

## 2.3.2 Backend AWS-Target Image Publish Contract

GitHub Actions is the canonical owner of backend AWS-target image publication.

Canonical workflow path in this repo:

- `/Users/chranama/career/llm-extraction-platform/.github/workflows/aws-image-publish.yml`

Current publication rules:

- build context:
  - repo root
- Dockerfile:
  - `/Users/chranama/career/llm-extraction-platform/deploy/docker/Dockerfile.server`
- publish platform:
  - `linux/amd64`
- publish target:
  - `ECR`
- canonical moving tags on the default branch:
  - `main`
  - `aws-dev-latest`
- canonical immutable tag:
  - `git-<sha>`
- later deploy consumers should prefer:
  - image digest
  - or the immutable `git-<sha>` tag

Credential contract:

- `vars.AWS_ROLE_TO_ASSUME` is the required GitHub-side input for OIDC-based publication
- `vars.AWS_REGION` is optional and defaults to `us-east-1`

The purpose of this slice is to make the backend image reviewer-reproducible and AWS-consumable without disturbing the existing local or `kind` paths.

## Backend Secrets And Config Contract

The backend side of the AWS slice should treat secrets and config as explicitly owned inputs.

Default first-slice contract:

- managed secret source:
  - `Secrets Manager`
- explicit runtime config wiring for:
  - database URL
  - Redis URL
  - API keys
  - observability settings
- Terraform-owned or otherwise explicitly documented materialization into the cluster

The goal is not a full secret-synchronization platform.
The goal is to keep the AWS slice reviewable and non-magical.

## Backend Kubernetes Overlay Contract

Canonical backend AWS overlay path:

- `/Users/chranama/career/llm-extraction-platform/deploy/k8s/overlays/aws-eks/`

This path is the AWS counterpart to:

- `/Users/chranama/career/llm-extraction-platform/deploy/k8s/overlays/local-observability-kind/`

At `2.3.1`, this is a scaffolded target path, not a full deployable overlay yet.

The backend AWS overlay is expected to become the home for:

- backend deployment deltas
- worker deployment deltas
- managed data connection wiring
- backend-specific secrets/config assumptions
- cloud log and observability assumptions that differ from local or `kind`

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

## 2.3.1 Acceptance Notes

`Phase 2.3.1` should now be treated as complete on the backend side only if the contract docs make explicit:

- how backend and worker logs participate in the bounded cloud inspection path
- how backend secrets and config are sourced and materialized
- how the backend participates in the smoke, inspection, and rollback/teardown workflow shape
- which named quality signals are allowed to influence runtime choice
- which seeded quality and `SLI` / `SLO` thresholds `Phase 2.3` should expose
- how the backend participates in minimal usage metering, quota or admission control, and rough cost attribution without adding a billing subsystem
- which proof artifacts support rollback, teardown, or keep decisions
