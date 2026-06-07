# AWS Deployment Contract

This document defines how `llm-extraction-platform` participates in the active
bounded AWS deployment for the integrated
`inference-serving-gateway` plus `llm-extraction-platform` stack.

The integrated AWS contract lives in the gateway repository:

- `/Users/chranama/career/inference-serving-gateway/docs/aws-deployment-contract.md`

This backend contract describes the backend API, async worker, image, config,
managed data, and proof expectations that the integrated deployment must
preserve.

## Status

Current AWS work is in progress.

Implemented or scaffolded surfaces:

- backend AWS-target image publication workflow at
  `.github/workflows/aws-image-publish.yml`;
- backend AWS/EKS overlay path at `deploy/k8s/overlays/aws-eks/`;
- local, Compose, joint, and `kind` proof paths that define the behavior AWS
  should preserve.

Still pending:

- deployable backend AWS overlay manifests;
- managed RDS/Redis secret and config wiring;
- AWS migration and proof-key seed jobs;
- AWS proof artifacts against the deployed ALB path.

## Backend Role In The AWS Stack

In AWS, the backend participates as:

- internal API service;
- async extraction worker;
- migration and seed-job owner;
- backend image owner;
- backend-specific Kubernetes overlay owner;
- source of usage, quota, trace, job, and rough-cost surfaces.

The public ingress path is gateway-led:

```text
ALB
  -> inference-serving-gateway
      -> llm-extraction-platform API
      -> llm-extraction-platform worker
      -> RDS PostgreSQL
      -> ElastiCache Redis
```

The backend API should not be directly exposed through public AWS ingress in the
first slice.

## Runtime Contract

The backend must preserve these semantics in AWS:

- `EDGE_MODE=behind_gateway`;
- gateway-owned request and trace identity;
- backend-generated async `job_id`;
- sync extract;
- async submit and status polling;
- admin/runtime trace inspection where enabled;
- usage and quota surfaces for the bounded proof key.

The external route contract through the gateway remains:

- `POST /v1/extract`;
- `POST /v1/extract/jobs`;
- `GET /v1/extract/jobs/{job_id}`.

`/v1/generate` is not a promoted AWS gateway route unless the gateway explicitly
adds that route later.

## Image Contract

GitHub Actions owns backend AWS-target image publication.

Workflow:

- `.github/workflows/aws-image-publish.yml`

Publication rules:

- build context: repository root;
- Dockerfile: `deploy/docker/Dockerfile.server`;
- platform: `linux/amd64`;
- ECR repository: `llm-server`;
- immutable tag: `git-<sha>`;
- dev moving tags: `main`, `aws-dev-latest`.

AWS deployment manifests should consume image digests or immutable `git-<sha>`
tags where practical. Local and `kind` paths should continue using local image
tags.

## Kubernetes Overlay Contract

Canonical backend AWS overlay path:

- `deploy/k8s/overlays/aws-eks/`

The overlay should provide backend-specific deltas from the local/kind shape:

- backend API Deployment;
- backend API internal Service;
- async worker Deployment;
- migration Job;
- proof-key or bounded usage seed Job;
- backend ConfigMaps;
- backend Kubernetes Secrets or explicit secret references;
- model/profile config appropriate for the first AWS proof;
- environment variables for managed RDS and Redis;
- OTel/logging configuration for cloud inspection.

The overlay should not reintroduce in-cluster Postgres or Redis for the AWS
slice. Managed data is part of the cloud proof.

## Managed Data Contract

The first AWS slice uses:

- RDS PostgreSQL for the backend database;
- ElastiCache Redis for async queue/state support.

Backend runtime inputs:

- `DATABASE_URL`;
- `REDIS_URL`;
- `REDIS_ENABLED=true`;
- database migration command;
- worker command.

Migration behavior must be explicit. The AWS path should not depend on an
operator remembering to run ad hoc database commands after deployment.

## Secrets And Config Contract

Required backend secrets/config:

- user/proof API key;
- admin API key;
- database URL or database credentials;
- Redis URL;
- optional model-provider credentials;
- OTel endpoint and service names;
- model config path/profile;
- schema directory.

Preferred managed source:

- AWS Secrets Manager.

Acceptable first-slice materialization:

- Terraform-owned Kubernetes Secret, if the secret source and rendered keys are
  documented.

The backend should avoid raw secret values in docs, logs, metrics, or proof
artifacts.

## Model Runtime Contract

The first AWS proof should start with a deterministic or low-variance backend
profile unless live model runtime is explicitly promoted.

Reason:

- the first AWS slice is proving cloud deployment, service boundaries, managed
  data, ingress, and inspection;
- live model quality and cloud model-serving acceleration can be added as a
  later proof target.

If a live model-backed AWS proof is added later, the contract must specify:

- model backend;
- compute requirements;
- provider or runtime credentials;
- expected latency budget;
- extract contract validation threshold;
- cost attribution assumptions.

## Identity And Trace Contract

Application identity remains authoritative in AWS:

- `request_id`;
- `trace_id`;
- `job_id`.

The backend should preserve these values in:

- database records;
- inference logs;
- admin trace endpoints;
- structured logs;
- proof artifacts.

AWS-native log or trace identifiers are supplementary. They do not replace the
application identifiers.

## Observability Contract

Backend API and worker logs should flow into the same bounded cloud log surface
as the gateway.

Default first-slice log surface:

- CloudWatch Logs.

Backend inspection should preserve:

- API logs for one sync extract;
- worker logs for one async extract;
- gateway/backend trace identity continuity;
- backend metrics snapshot;
- usage or rough-cost snapshot tied to the same proof key.

In-cluster Jaeger and Prometheus/Grafana are acceptable for the first AWS proof
if documented as bounded, session-oriented components.

## Usage, Quota, And Cost Contract

The first bounded usage scope is:

- `api_key`.

Backend-owned metering surfaces:

- `GET /v1/me/usage`;
- `GET /v1/admin/usage`;
- inference logs with token and latency data;
- API-key quota state;
- proof artifacts.

The first AWS proof should include one visible fairness or spend-control rule.
The preferred path is to reuse existing API-key quota or backend admission
behavior rather than introduce a billing subsystem.

Raw API keys must not become default Prometheus labels.

## Backend Proof Contract

The backend side of the AWS proof should make these artifacts possible:

- one sync extract producing contract-valid structured output;
- one async job that reaches terminal success through the worker;
- one correlated trail across gateway, backend API, worker, and database state;
- one backend metrics snapshot;
- one usage or rough-cost snapshot tied to the proof key;
- one note showing whether repair, policy, or quota affected the proof request.

The proof should be generated through the gateway ALB path, not by directly
calling the backend Service from inside the cluster.

## Rollback Or Teardown Criteria

Backend-driven rollback or teardown is justified when:

- sync extract stops producing contract-valid output;
- async jobs stop completing within the bounded timeout;
- database migrations fail or drift from expected schema state;
- request identity cannot be followed through backend/worker logs;
- quota/admission behavior blocks the proof path unexpectedly;
- rough request cost changes materially without an intentional runtime change.

For the first bounded slice, teardown with `terraform destroy` remains an
acceptable operator path.

## Acceptance Criteria

The backend AWS contract is satisfied when:

- the backend AWS image can be published to ECR;
- backend API and worker run on EKS using the AWS image;
- backend uses RDS PostgreSQL and ElastiCache Redis;
- migrations run explicitly;
- a proof key or bounded usage scope is seeded explicitly;
- sync extract succeeds through the gateway ALB path;
- async extract succeeds through the gateway ALB path and worker;
- backend logs, metrics, traces, and usage surfaces are inspectable;
- proof artifacts match the deployed AWS behavior;
- no backend public ingress bypasses the gateway in the first slice.

## Implementation Gaps To Close Next

1. Finish or commit the backend AWS image publication workflow.
2. Replace the AWS overlay README-only scaffold with deployable manifests.
3. Define the RDS/Redis secret materialization path.
4. Add migration and proof-key seed jobs for AWS.
5. Integrate backend proof capture into the joint AWS proof harness.
