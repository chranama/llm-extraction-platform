# AWS EKS Backend Overlay

This directory is the canonical backend-specific Kubernetes overlay path for the AWS/EKS deployment slice.

Planning reference:

- `/Users/chranama/career/job-search/audit/2026-03-28__phase2-3-aws-deployment-slice-implementation-plan.md`

Backend-side AWS contract:

- `/Users/chranama/career/llm-extraction-platform/docs/aws-deployment-contract.md`

This path now carries the backend-owned AWS/EKS overlay for the first bounded AWS
slice. The gateway repository owns the integrated front door, ALB ingress,
Terraform substrate, and joint AWS runbook.

## Contract Responsibilities

This path should eventually carry the backend-specific AWS/EKS deltas needed to preserve:

- backend API and worker identity continuity across:
  - `request_id`
  - `trace_id`
  - `job_id`
- explicit managed-data wiring for `RDS PostgreSQL` and `ElastiCache Redis`
- explicit secrets and config wiring
- the backend-side runtime-quality signals used for cloud-slice judgment
- the bounded usage, quota, and rough-cost surfaces that remain authoritative in the backend

## Minimum 2.3.1 Proof Inputs

Even before manifests land, this path should make room for:

- one sync extract proof that reaches valid structured output
- one async proof that completes within the bounded timeout budget
- one backend or worker correlation trail in logs and traces
- one usage or rough-cost snapshot tied to the same bounded `api_key` scope
- one note about whether repair or policy influenced the result

## Contents

- `kustomization.yaml`: backend AWS overlay.
- `models.aws-proof.yaml`: deterministic model profile for cloud deployment proof.
- `server-patch.yaml`: API runtime settings for gateway mode, tracing, and model loading.
- `worker-deployment.yaml`: async extract worker.
- `migrations-patch.yaml`: explicit RDS-backed migration command.
- `seed-proof-keys-job.yaml`: proof user/admin key seeding with a bounded user quota.
- delete patches for local-only Postgres, Redis, ingress, static Secret, and policy gate.

Runtime secrets are materialized by the gateway repository's AWS harness at
deploy time. Live RDS, Redis, and API-key values are not committed in this
overlay.
