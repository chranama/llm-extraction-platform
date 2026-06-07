# AWS EKS Backend Overlay

This directory is the canonical backend-specific Kubernetes overlay path for the AWS/EKS deployment slice.

Planning reference:

- `/Users/chranama/career/job-search/audit/2026-03-28__phase2-3-aws-deployment-slice-implementation-plan.md`

Backend-side AWS contract:

- `/Users/chranama/career/llm-extraction-platform/docs/aws-deployment-contract.md`

At `2.3.1`, this path is still scaffold-only, but the scaffold now has to preserve the runtime-quality and usage contracts defined before cloud rollout.

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

## Expected Future Contents

- backend API AWS overlay
- managed data connection wiring
- AWS-target config/secrets assumptions
- backend deployment deltas relative to the local and `kind` paths
- worker deployment deltas relative to the local and `kind` paths
- cloud log-path assumptions for backend and worker correlation
- backend-side participation in the smoke and inspection workflow
- backend-side surfacing of quota, admission, or fairness controls without adding a billing subsystem

## 2.3.1 Done Means

This path does not need full manifests yet.
It does need to make the backend-side proof and runtime-quality assumptions explicit enough that later overlays can be judged against them.
