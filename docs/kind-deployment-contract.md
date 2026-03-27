# 14) Phase 2 Kind Deployment Contract

This document mirrors the gateway-side Phase 2 kind contract and defines how `llm-extraction-platform` participates in the Kubernetes-shaped local stack.

Gateway-side reference:

- `/Users/chranama/career/inference-serving-gateway/docs/kind-deployment-contract.md`

## Purpose

The kind path is the Kubernetes complement to the host-run local Phase 2 stack.

It exists so the integrated system can be demonstrated in two realistic modes:

- a fast local orchestration path
- a cluster-shaped deployment path

## Canonical backend overlay

The backend participates through:

- `/Users/chranama/career/llm-extraction-platform/deploy/k8s/overlays/local-observability-kind`

That overlay is responsible for:

- reusing the backend Kubernetes base
- mounting an observability-proof models config
- running the API with:
  - `APP_PROFILE=test`
  - `MODELS_PROFILE=observability-proof`
  - `EDGE_MODE=behind_gateway`
- disabling nonessential local policy-gate behavior for the kind demo

## Canonical models config

The kind overlay uses a self-contained copy of the observability-proof models file:

- `/Users/chranama/career/llm-extraction-platform/deploy/k8s/overlays/local-observability-kind/models.observability-proof.yaml`

This keeps the overlay reproducible under Kustomize without depending on files outside the overlay root.

## Backend assumptions

The kind path assumes:

- in-cluster Postgres service name: `postgres`
- in-cluster Redis service name: `redis`
- API deployment name: `api`
- API service name: `api`

The gateway service should be the only trusted source of injected trace IDs, which is why the backend overlay runs with:

- `EDGE_MODE=behind_gateway`

## Integrated extras supplied from the gateway repo

The backend overlay does not include:

- the async worker deployment
- the gateway deployment/service
- the proof-key seed job

Those integrated stack resources live in:

- `/Users/chranama/career/inference-serving-gateway/deploy/k8s/local-kind-stack`

This split keeps backend ownership focused on backend infrastructure while letting the gateway repo remain the canonical front door for integrated stack orchestration.
