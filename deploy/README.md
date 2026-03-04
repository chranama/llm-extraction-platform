# Deploy

## Purpose
Deployment assets for compose, docker images, k8s overlays, observability, and proxying.

## Key Entrypoints
- `deploy/compose/docker-compose.yml`
- `deploy/docker/`
- `deploy/k8s/`

## Run/Test
```bash
uv run llmctl --project-name llmep compose --env-override-file .env.docker ps
```

## Dependencies
- Profiles consume `config/` and run services from `server/`, `ui/`, and infra.

## Deep Links
- [`/docs/03-deployment-modes.md`](../docs/03-deployment-modes.md)
