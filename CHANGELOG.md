# Changelog
All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)  
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]
### Added
- (placeholder) Add new features here after 1.0.0.

### Changed
- (placeholder)

### Fixed
- (placeholder)

---

## [1.0.0] – 2025-01-XX
### Added
#### Core System
- Initial release of **LLM Server** — a production-style LLM API gateway and inference runtime.
- FastAPI-powered API gateway with:
  - Authentication (API keys)
  - Role system including `admin`
  - Quotas (monthly) and usage tracking
  - RPM (rate limit) enforcement
  - Request logging
  - Token counting (approximate input/output)
- Full health checks: `/healthz` and `/readyz`.

#### Model Execution
- Local inference runtime supporting:
  - **Apple MPS** (local mode default)
  - CPU execution
- Dockerized CPU inference runtime.
- Default models:
  - Local mode: **Llama 3.1 8B**
  - Container mode: **Llama 3.2 1B**
- MultiModelManager abstraction with support for:
  - Local models
  - Remote HTTP-based LLM backends

#### Caching
- Deduplicating response cache stored in Postgres (CompletionCache).
- Cache key includes `model_id`, prompt hash, and generation params.
- Redis hot-cache integration (optional) with feature flag.

#### Observability & Ops
- Prometheus metrics endpoint (`/metrics`)
- Built-in metrics for:
  - Request latency
  - Queue depth
  - Cache hits/misses
  - RPM limit events
- Grafana dashboards:
  - LLM API Overview
  - Prometheus system view
- Structured JSON logging.
- pgAdmin container for database access.

#### UI
- Lightweight React/Vite playground at `/ui`.
- API key injection system for browser clients.
- Nginx reverse proxy exposing:
  - `/api/*`
  - `/healthz`
  - `/readyz`
  - Admin dashboards under `/grafana`, `/prometheus`, `/pgadmin`.

#### Database & Migrations
- Alembic migration system with:
  - Roles table
  - API keys table
  - Request logs table
  - Completion cache table
- Automated migration workflow via Makefile.

#### Developer Tooling
- Full Makefile automation:
  - `dev-local`, `dev-cpu`, `up`, `down`, `restart`
  - Logs and debugging helpers
  - Database migration helpers
  - API key seeding and inspection
- Environment-based configuration via `.env` and `.env.local`.
- uv-based dependency management.

### Changed
- N/A — initial release.

### Fixed
- N/A — initial release.

---

## [0.1.0] – 2024-XX-XX (Prehistory)
_This section is optional; include only if you want historical context._

- Early experimental scripts for hosting a local Mistral model.
- First prototypes of API routing logic.
- Observability experiments with Prometheus and Grafana.
- Authentication prototyping.