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
- deterministic backend AWS/EKS overlay path at `deploy/k8s/overlays/aws-eks/`;
- live vLLM backend AWS/EKS overlay path at
  `deploy/k8s/overlays/aws-eks-vllm/`;
- local, Compose, joint, and `kind` proof paths that define the behavior AWS
  should preserve.

Still pending:

- live RDS/Redis secret materialization from the gateway-owned AWS harness;
- execution of AWS migration and proof-key seed jobs against EKS;
- AWS fake-backend proof artifacts against the deployed ALB path;
- AWS vLLM proof artifacts against the deployed ALB path.

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
      -> optional vLLM model runtime
      -> RDS PostgreSQL
      -> ElastiCache Redis
```

The backend API should not be directly exposed through public AWS ingress in the
first slice.

The backend supports two promoted AWS profiles:

| Workflow | Overlay | Model Runtime |
|---|---|---|
| `AWS_WORKFLOW=fake` | `deploy/k8s/overlays/aws-eks/` | deterministic in-process fake backend |
| `AWS_WORKFLOW=vllm` | `deploy/k8s/overlays/aws-eks-vllm/` | external OpenAI-compatible vLLM Service inside the cluster |

## Backend AWS Component Inventory

The backend participates in this AWS component set:

| Component | What It Does For The Backend | First-Slice Posture |
|---|---|---|
| ECR | Stores the CI-built backend image used by the API, worker, migration, and seed-job pods | `llm-server`, published as `linux/amd64` |
| EKS | Runs the backend Kubernetes workloads in the joint stack | Backend pods are internal behind the gateway |
| EC2 managed node group | Supplies the compute where backend pods are scheduled | Shared with gateway and observability pods |
| EC2 GPU managed node group | Supplies accelerated compute for the vLLM model-runtime pod | Disabled unless `AWS_WORKFLOW=vllm` is selected |
| RDS PostgreSQL | Provides the managed database for API keys, logs, traces, jobs, and usage state | Managed replacement for in-cluster Postgres |
| ElastiCache Redis | Provides managed Redis for async extraction queue/state behavior | Managed replacement for in-cluster Redis |
| Secrets Manager | Stores backend API keys and connection secrets outside source control and manifests | May be materialized into Kubernetes Secrets for the first slice |
| CloudWatch Logs | Collects backend API and worker logs for cloud-side inspection | Must preserve `request_id`, `trace_id`, and `job_id` |
| IAM | Defines image-publish, image-pull, and future workload permission boundaries | Small explicit role set |
| VPC and security groups | Provide network reachability and firewall boundaries between backend pods and managed data | Backend access scoped to the bounded VPC |
| Application Load Balancer | Provides public ingress to the gateway; backend traffic arrives only after gateway routing | Backend is reached only through the gateway route path |

In-cluster OTel Collector, Jaeger, Prometheus, and Grafana remain part of the
reviewable AWS proof path, but they are not backend-owned AWS managed services.

The vLLM model runtime uses a separate public image, `vllm/vllm-openai`. That
image is not the backend API image and should not be pulled into the API,
worker, migration, or seed-job pods.

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

The promoted AWS `llm-server` image is a slim application image. Default
dependencies must not include Torch, Transformers, llama.cpp, or other
model-serving stacks. Those dependencies are allowed only behind explicit local
or legacy extras, while the vLLM workflow runs model serving in its own
Deployment.

## Kubernetes Overlay Contract

Canonical backend AWS overlay paths:

- `deploy/k8s/overlays/aws-eks/` for `AWS_WORKFLOW=fake`;
- `deploy/k8s/overlays/aws-eks-vllm/` for `AWS_WORKFLOW=vllm`.

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

The vLLM overlay additionally provides:

- vLLM Deployment and Service;
- `backend: vllm` model profile;
- OpenAI-compatible remote backend configuration;
- GPU resource request and node scheduling constraints;
- optional Hugging Face token materialization when the selected model requires
  authentication.

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
- optional Hugging Face token for the vLLM workflow;
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

The backend promotes two AWS model-runtime profiles:

| Profile | Runtime | Compute | Purpose |
|---|---|---|---|
| `aws-fake` | deterministic in-process fake backend | CPU node group | Prove cloud deployment, routing, state, policy, traces, and teardown without model variance |
| `aws-vllm` | external OpenAI-compatible vLLM Service | GPU node group with `nvidia.com/gpu: 1` | Prove live model serving through the same gateway/backend extract path |

The vLLM profile treats model serving as an external runtime from the backend
API's point of view. The backend reads an OpenAI-compatible endpoint, readiness
probe, model name, and request mode from model config, then sends extract and
generate traffic through the remote backend adapter.

The selected vLLM model for the first workflow should be small enough for a
single bounded AWS GPU node. The initial profile uses Qwen/Qwen3-0.6B and a
bounded context length. Any larger model requires an explicit contract update
covering instance type, expected latency, extract contract behavior, and cost.

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
- vLLM logs and metrics for the live model workflow;
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
- one vLLM log and metrics snapshot when `AWS_WORKFLOW=vllm`;
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

For `AWS_WORKFLOW=vllm`, the backend contract also requires:

- the vLLM Deployment becomes ready on a GPU node;
- backend model readiness reflects the remote vLLM endpoint;
- extract traffic reaches vLLM through the backend remote adapter;
- vLLM logs and metrics are captured with the backend proof artifacts.

## Implementation Gaps To Close Next

1. Publish the backend image into ECR.
2. Render and apply the deterministic backend overlay through the gateway-owned
   AWS harness.
3. Run migrations and proof-key seed jobs against RDS.
4. Capture backend logs, metrics, usage, and trace evidence for
   `AWS_WORKFLOW=fake`.
5. Enable the vLLM overlay through `AWS_WORKFLOW=vllm` and capture the same
   backend proof plus vLLM logs and metrics.
