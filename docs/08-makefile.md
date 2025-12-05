# Makefile

The Makefile provides a standardized developer and operator experience for managing the LLM Server.  
It wraps common workflows such as infrastructure startup, migrations, key seeding, and debugging.

This section documents:

- Golden-path usage
- Local (MPS/host) development mode
- Container control and logs
- Database migrations
- Quick API tests
- Cleanup operations

---

## 1. Golden Path Targets (CPU / Container Mode)

These are the recommended commands for most users.

### make init
Ensures a `.env` file exists.

Behavior:
- If `.env` exists → keep it.
- If `.env` does not exist but `.env.example` exists → copy `.env.example` to `.env`.
- If neither exists → abort with an error.

Example:
    make init

---

### make up
Starts the full **CPU container stack**:

- Postgres
- Redis
- API container (LLM + FastAPI)
- Prometheus
- Grafana
- pgAdmin
- Nginx (public entrypoint)

After running:
- UI: http://localhost:8080/ui/
- API (proxied): http://localhost:8080/api/v1/generate
- API (direct):  http://localhost:8000/v1/generate
- Prometheus: http://localhost:8080/prometheus/
- Grafana:    http://localhost:8080/grafana/
- pgAdmin:    http://localhost:8080/pgadmin/

Run:
    make up

---

### make seed-key API_KEY=<your-key>
Seeds an admin API key into Postgres.

Example:
    make seed-key API_KEY=$(openssl rand -hex 24)

Use this key in:
- curl → header `X-API-Key: <your-key>`
- UI → automatically provided via build-time arg when using Docker

---

### make curl API_KEY=<your-key>
Quick sanity test via Nginx using a sample prompt.

Example:
    make curl API_KEY=<your-key>

---

### make down
Stops containers but **keeps data volumes**.

    make down

---

### make clean
Stops containers and removes orphans, preserving volumes.

    make clean

---

### make nuke
Dangerous reset: removes all containers, networks, and optionally volumes belonging to this project.

Use only when Compose state becomes inconsistent.

    make nuke

---

## 2. Local (MPS / Host) Development Targets

### make dev-local
Uses `.env.local` to run API on host (MPS) and infra in Docker.

This starts:
- postgres
- redis
- prometheus
- grafana
- pgadmin
- nginx

Then you separately run the API with `make api-local`.

Run:
    make dev-local

---

### make api-local
Runs FastAPI on host using local MPS settings.

Example:
    ENV_FILE=.env.local make api-local

---

### make up-local
Starts infra-only containers for local dev.

    make up-local

---

### make dev-tmux
Opens tmux with:
- Pane 1: `make api-local`
- Pane 2: `make logs-nginx`

    make dev-tmux

---

## 3. Container Control & Logs

### make ps
Show container status:
    make ps

### make logs
Tail all container logs:
    make logs

### make logs-nginx
Tail only Nginx logs:
    make logs-nginx

### make logs-postgres
Tail Postgres logs:
    make logs-postgres

### make logs-redis
Tail Redis logs:
    make logs-redis

---

## 4. Database Migrations

### make migrate
Apply Alembic migrations on host using ENV_FILE:
    ENV_FILE=.env.local make migrate

### make migrate-docker
Apply migrations **inside API container**:
    make migrate-docker

### make revision
Autogenerate migration:
    make revision m="add new column"

---

## 5. Quick API Tests

### make test API_KEY=<your-key>
Sends “ping” request directly to API and Nginx.

Example:
    make test API_KEY=<your-key>

---

## 6. Environment Inspection

### make env
Print effective environment variables loaded from ENV_FILE.

Example:
    make env

---

## 7. Developer Mode for CPU

### make dev-cpu
Same as `make up`, but also runs migrations and prints guidance for testing.

    make dev-cpu

---

## Summary Workflow

Most users should run:

    make init
    make up
    make seed-key API_KEY=$(openssl rand -hex 24)
    make curl API_KEY=<your-key>

