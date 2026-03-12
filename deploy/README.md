# Deploy

## Purpose
Deployment assets for compose, docker images, k8s overlays, observability, and proxying.

## Key Entrypoints
- `deploy/compose/docker-compose.yml`
- `deploy/docker/`
- `deploy/k8s/`

## Kubernetes Reviewer Path
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

## Proof Boundary
- Local `kind` proof demonstrates runnable Kubernetes deployment.
- `prod-gpu-full` render demonstrates scaffold readiness only.
- This surface does not claim real GPU scheduling or production-scale operation.

## Deep Links
- [`/docs/03-deployment-modes.md`](../docs/03-deployment-modes.md)
