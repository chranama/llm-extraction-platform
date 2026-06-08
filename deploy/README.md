# Deploy

## Purpose
Deployment assets for compose, docker images, k8s overlays, observability, and proxying.

## Key Entrypoints
- `deploy/compose/docker-compose.yml`
- `deploy/docker/`
- `deploy/k8s/`
- `.github/workflows/aws-image-publish.yml`

## Kubernetes Local Flow
1. Bring up `kind`
2. Build and load the server image
3. Apply `deploy/k8s/overlays/local-generate-only`
4. Wait for rollout
5. Run `tools/k8s/k8s_smoke.sh`
6. Inspect `proof/artifacts/phase5_k8s_kind/`

## Run/Test
```bash
uv run llmctl --project-name llmep compose --env-override-file .env.docker ps
```

## Dependencies
- Profiles consume `config/` and run services from `server/`, `ui/`, and infra.

## Scope Boundary
- Local `kind` evidence shows runnable Kubernetes deployment.
- This surface does not claim real GPU scheduling or production-scale operation.

## AWS Image Publish Path
- `Phase 2.3.2` makes the backend AWS image CI-owned through `.github/workflows/aws-image-publish.yml`.
- The canonical AWS-target Dockerfile remains `deploy/docker/Dockerfile.server`.
- Later AWS deploy steps should consume the published digest or `git-<sha>` tag rather than local-only image tags.

## Async Extract Local Runtime
- The async extraction evidence runs outside Kubernetes in a local host-runtime layout.
- API server and worker run as separate processes.
- Redis carries queue delivery; Postgres remains the source of truth for job state.
- Canonical artifacts land in `proof/artifacts/phase6_extract_async/`.

## Deep Links
- [`/docs/operations.md`](../docs/operations.md)
