## Observability

The stack ships with a full observability setup, but in “production-style” mode only
**nginx is exposed to the host**. Prometheus, Grafana, and pgAdmin run on the internal
Docker network and are only reachable through nginx.

If you want direct container access on the host (e.g., `localhost:3000`), use a
development compose file such as `docker-compose.dev.yml` with extra `ports:` mappings.

### 1. API Health & Metrics

The FastAPI gateway exposes basic health and metrics endpoints:

- `GET /healthz` – lightweight liveness probe
- `GET /readyz` – readiness probe (DB + Redis + model checks)
- `GET /metrics` – Prometheus metrics endpoint

These are served by the API container but typically fronted by nginx. In the default
topology you reach them via:

- `http://localhost:8080/api/healthz`
- `http://localhost:8080/api/readyz`
- `http://localhost:8080/api/metrics`

### 2. Prometheus (via nginx)

Prometheus runs on the internal Docker network and is **not** bound directly to a host
port in the production-style compose file. nginx proxies it under a path prefix.

Public entrypoint (through nginx):

- `http://localhost:8080/prometheus/`

Prometheus is configured with:

- `--web.route-prefix=/prometheus`
- `--web.external-url=http://localhost:8080/prometheus`

So all links and redirects stay under `/prometheus/...` when accessed through nginx.

Typical things you might do:

- Explore metrics in the Prometheus UI
- Inspect scraped targets
- Debug custom application metrics from `/metrics`

### 3. Grafana (via nginx)

Grafana is also internal-only and accessed via nginx. In production-style mode there is
**no** direct `localhost:3000` binding; instead you go through the reverse proxy.

Public entrypoint:

- `http://localhost:8080/grafana/`

Grafana is configured with:

- `GF_SERVER_ROOT_URL=http://localhost:8080/grafana/`
- `GF_SERVER_SERVE_FROM_SUB_PATH=true`

This ensures Grafana correctly serves under the `/grafana/` sub-path and does not try
to redirect to its internal port.

Example ready-to-use dashboards might include:

- API latency and error rates
- Model throughput and token usage
- Cache hit rates
- System-level metrics sourced from Prometheus

### 4. pgAdmin (via nginx)

pgAdmin provides a browser-based UI for Postgres and is also served through nginx in
production-style mode.

Public entrypoint:

- `http://localhost:8080/pgadmin/`

Key notes:

- pgAdmin is configured with `SCRIPT_NAME=/pgadmin` so it knows it is mounted under
  that sub-path.
- Use the Postgres service name (e.g., `llm_postgres`) and internal port (`5432`)
  when registering a connection in pgAdmin.
- Defaults (if you haven’t changed them) are controlled by the `PGADMIN_*` environment
  variables in the compose file.

### 5. Public vs Internal Ports

The intent of the production-style compose is:

- **Public-facing ports**:
  - `8080` – nginx (UI + API + observability proxies)

- **Internal-only services** (no host `ports:` in the production compose):
  - Prometheus (`prometheus:9090`)
  - Grafana (`grafana:3000`)
  - pgAdmin (`pgadmin:80`)
  - Postgres (`llm_postgres:5432`)
  - Redis (`llm_redis:6379`)

All access from the outside world flows through nginx on `localhost:8080`, where:

- `/ui/`        → React playground
- `/api/...`    → FastAPI gateway
- `/prometheus/`→ Prometheus UI
- `/grafana/`   → Grafana UI
- `/pgadmin/`   → pgAdmin UI

### 6. Development vs Production

For local development you may want direct container ports (e.g., `localhost:3000` for
Grafana, `localhost:9090` for Prometheus, `localhost:5050` for pgAdmin).

Recommended workflow:

1. **Production-style file** (`docker-compose.yml`)
   - Only nginx is exposed.
   - Observability is reached via path prefixes on `:8080`.

2. **Development file** (`docker-compose.dev.yml`)
   - Extends or copies the base file.
   - Adds explicit `ports:` mappings such as:
     - `9090:9090` for Prometheus
     - `3000:3000` for Grafana
     - `5050:80` for pgAdmin

This keeps the main compose file aligned with a hardened, reverse-proxy-only surface,
while the dev file remains convenient for debugging and experimentation.