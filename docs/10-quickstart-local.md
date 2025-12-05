# Quickstart: Local Development (MPS / CPU on Host)

This section describes how to run **LLM Server locally on your machine** without Docker hosting the model.  
This mode is ideal for **Apple Silicon (MPS)** or **CPU-only** experimentation, fast iteration, and development.

In **local mode**:

- The **API server + LLM** run directly on your machine.
- Docker Compose runs **only the infrastructure**:
  - Postgres
  - Redis
  - Prometheus
  - Grafana
  - Nginx (public entrypoint)
  - pgAdmin

This hybrid design gives you the fastest development loop while maintaining a realistic production environment.

---

## 1. Setup Environment

Local mode uses `.env.local`. Create it automatically:

    make init

This produces:

- `.env`
- `.env.local` (configured for MPS or CPU execution on host)

You may edit `.env.local` to choose a model or change configuration.

---

## 2. Start Infrastructure Services

Start Docker-hosted services:

    make up-local

This launches:

- postgres  
- redis  
- prometheus  
- grafana  
- pgadmin  
- nginx  

The API server is **not** started yet (that happens next).

---

## 3. Run the API Locally

In a separate terminal:

    ENV_FILE=.env.local make api-local

This command:

- Loads configuration from `.env.local`
- Runs FastAPI + model on your host
- Binds to:  
  http://127.0.0.1:8000

---

## 4. Seed an API Key

Even in local mode, the API requires authentication.

Seed a key into the Postgres container:

    make seed-key API_KEY=$(openssl rand -hex 24)

Use the provided API key in requests via:

    X-API-Key: <your-key>

---

## 5. Test the API

### Direct API (bypassing Nginx)

    curl -X POST http://127.0.0.1:8000/v1/generate \
      -H "Content-Type: application/json" \
      -H "X-API-Key: <your-key>" \
      -d '{ "prompt": "ping", "max_new_tokens": 4 }'

### Through Nginx (recommended)

    curl -X POST http://localhost:8080/api/v1/generate \
      -H "Content-Type: application/json" \
      -H "X-API-Key: <your-key>" \
      -d '{ "prompt": "ping", "max_new_tokens": 4 }'

---

## 6. Optional: tmux Auto-Setup

If you want a fully managed developer session:

    make dev-tmux

This starts:

- API in one pane  
- Nginx logs in another pane  
- All infra already running  

Requires `tmux` installed.

---

## 7. Stop or Reset

### Stop infrastructure (keep DB + Redis data)

    make down

### Stop & remove containers (keep volumes)

    make clean

### Full reset including volumes (DESTRUCTIVE)

    make nuke

---

## Summary

The standard development workflow is:

    make init
    make up-local
    ENV_FILE=.env.local make api-local
    make seed-key API_KEY=$(openssl rand -hex 24)
    make curl API_KEY=<your-key>

Local mode gives you:

- **Fast iteration** (no rebuilding API container)
- **MPS acceleration** on Apple Silicon
- **Full observability stack**
- **Same auth, quotas, and caching** as production
