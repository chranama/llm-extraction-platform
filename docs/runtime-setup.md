# Runtime Setup

This document describes the setup behind the promoted local workflow and the
bounded AWS deployment workflows.

## Promoted Local Setup

The promoted local path is the joint `kind` deployment with a live CPU-only
`llama.cpp` model server:

- `llm-extraction-platform` API and async worker run in a local Kubernetes
  cluster.
- `inference-serving-gateway` runs in the same cluster and forwards extraction
  requests to the API.
- `llama-server` runs in the cluster and serves a host-mounted GGUF model.
- Postgres, Redis, OTel Collector, and Jaeger run as cluster resources.

This path is meant for reviewer inspection and technical interview
presentation. It demonstrates the service boundary, Kubernetes resources,
gateway forwarding, async jobs, traces, and live model-backed extraction without
generating proof artifacts.

## Local Hardware Requirements

Recommended local machine:

- Apple Silicon Mac or comparable developer machine.
- Docker Desktop with at least 8 CPU cores and 12-16 GB memory available.
- Enough disk space for Docker images and the GGUF model file.
- `kind`, `kubectl`, Docker, Go, Python, and `uv`.

The local live-model path is CPU-only. Use:

```bash
LLAMA_N_GPU_LAYERS=0
```

Docker Desktop on macOS runs Linux containers inside a VM. The promoted local
kind path does not depend on Metal/MPS or GPU passthrough. It is optimized for
reviewability and portability, not throughput.

## Local Model Configuration

Create `.env.docker` in the `llm-extraction-platform` checkout:

```bash
API_KEY=<local-api-key>
LLAMA_MODELS_DIR=/absolute/path/to/gguf-model-directory
LLAMA_MODEL_FILE=/models/path/inside/mounted-directory.gguf
LLAMA_N_GPU_LAYERS=0
```

`LLAMA_MODELS_DIR` is mounted into the kind control-plane node at `/models`.
`LLAMA_MODEL_FILE` must point to the model path as seen inside that mount.

If the `llm` kind cluster already exists and was created before the model mount
was configured, recreate it:

```bash
kind delete cluster --name llm
```

Then rerun the promoted kind startup command.

## AWS Setup

The AWS workflows are bounded deployment proofs. They create resources at
runtime and should be destroyed after smoke and inspection.

Required tools and access:

- AWS account with access to the console.
- AWS CLI configured through SSO, for example `AWS_PROFILE=llm-dev`.
- Terraform.
- `kubectl`, `helm`, `eksctl`, Docker, and GitHub CLI.
- GitHub Actions OIDC provider in AWS for image publishing.

The Terraform substrate lives in the gateway repository:

```text
inference-serving-gateway/deploy/aws/terraform/environments/dev/
```

The harness is:

```text
inference-serving-gateway/proof/run_aws_stack.sh
```

## AWS Workflow Modes

The fake backend workflow proves the cloud service boundary with deterministic
model behavior:

```bash
AWS_PROFILE=llm-dev AWS_WORKFLOW=fake proof/run_aws_stack.sh preflight
AWS_PROFILE=llm-dev AWS_WORKFLOW=fake proof/run_aws_stack.sh terraform-plan
AWS_PROFILE=llm-dev AWS_WORKFLOW=fake proof/run_aws_stack.sh terraform-apply
AWS_PROFILE=llm-dev AWS_WORKFLOW=fake proof/run_aws_stack.sh deploy
AWS_PROFILE=llm-dev AWS_WORKFLOW=fake proof/run_aws_stack.sh smoke
AWS_PROFILE=llm-dev AWS_WORKFLOW=fake proof/run_aws_stack.sh inspect
AWS_PROFILE=llm-dev AWS_WORKFLOW=fake proof/run_aws_stack.sh delete-workloads
AWS_PROFILE=llm-dev AWS_WORKFLOW=fake proof/run_aws_stack.sh delete-addons
AWS_PROFILE=llm-dev AWS_WORKFLOW=fake proof/run_aws_stack.sh terraform-destroy
```

The vLLM workflow adds a GPU node group and a live OpenAI-compatible model
runtime:

```bash
AWS_PROFILE=llm-dev AWS_WORKFLOW=vllm TF_VAR_enable_gpu_node_group=true \
  proof/run_aws_stack.sh preflight
```

The vLLM path requires the EC2 quota named `Running On-Demand G and VT
instances` to cover the selected GPU instance. The promoted g6.xlarge workflow
requires at least 4 vCPUs. Request 8 vCPUs if you want room for one active node
and a small safety margin.

## AWS Prep Checklist

- AWS SSO profile can run `aws sts get-caller-identity`.
- Terraform initializes and validates in the dev environment.
- GitHub Actions OIDC provider exists for `token.actions.githubusercontent.com`.
- Ephemeral ECR repositories are created by Terraform and destroyed by
  Terraform teardown.
- The GPU quota is approved before running `AWS_WORKFLOW=vllm`.
- The NVIDIA device plugin is installed for the vLLM workflow.
- The AWS Load Balancer Controller is installed before public ALB smoke.
- `terraform-destroy` is run after each proof session.

## Cost Posture

The AWS setup is intentionally ephemeral. It is not a standing production
environment. The expensive resources are the EKS control plane, GPU node group,
RDS, Redis, ALB, and stored container images. The workflow should end with:

```bash
AWS_PROFILE=llm-dev AWS_WORKFLOW=<fake|vllm> proof/run_aws_stack.sh terraform-destroy
```

