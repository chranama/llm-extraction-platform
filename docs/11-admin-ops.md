# Admin & Operations Guide

This section documents the administrative and operational APIs available in **LLM Server**.  
These endpoints enable operators to view usage, inspect logs, manage API keys, and monitor system health.

**Important:**  
Admin endpoints require an API key associated with the `admin` role, which can be created using:

    make seed-key API_KEY=<value>

---

## 1. Usage & Quotas

### GET `/v1/me/usage` — *Stable*

Returns usage information for the authenticated API key.

**Example**

    curl http://localhost:8080/api/v1/me/usage \
      -H "X-API-Key: <your-key>"

**Response fields include (shape may grow over time):**

- `quota_used`
- `quota_monthly`
- `quota_reset_at`
- `rpm_limit`      – rate-limit for this key
- `rpm_remaining`  – remaining allowance in the current window

Use cases:

- Showing usage in a UI
- Building simple billing/reporting
- Monitoring heavy users

---

## 2. API Key Management

These endpoints allow administrative control over authentication credentials.

All of them require an admin key.

### GET `/v1/admin/keys` — *Stable*

List API keys, their labels, roles, status, and usage summary.

    curl http://localhost:8080/api/v1/admin/keys \
      -H "X-API-Key: <admin-key>"

### POST `/v1/admin/keys` — *Experimental*

Programmatically create a new API key.

Body example (conceptual):

    {
      "label": "client-a",
      "role": "default"
    }

### PATCH `/v1/admin/keys/{key}` — *Experimental*

Update an existing key (for example, enable/disable or change role).

### DELETE `/v1/admin/keys/{key}` — *Experimental*

Revoke an API key.

> **Why experimental?**  
> The exact request/response format and filtering options may change in a future minor release as admin workflows are refined.

---

## 3. Log Access

LLM Server can log inference activity into Postgres, including:

- request metadata
- prompt and response snippets (redaction strategy is up to the operator)
- latency
- cached vs non-cached
- error codes

### GET `/v1/admin/logs` — *Experimental*

Fetch recent logs from the `inference_logs` table.

Potential usage patterns (syntax indicative, not guaranteed stable):

- `/v1/admin/logs?limit=100`
- `/v1/admin/logs?model=mistral`
- `/v1/admin/logs?cached=true`

Use this for:

- debugging production-like workloads
- investigating high latency or failures
- building internal dashboards

Because the shape and filters are likely to evolve, this endpoint is marked **experimental**.

---

## 4. System Health & Metrics

These endpoints are safe to expose internally without admin keys.

### GET `/healthz` — *Stable*

Simple liveness probe.

- Returns `200 OK` if the API process is running.
- Intended for container orchestrators and basic uptime checks.

### GET `/readyz` — *Stable*

Readiness probe that confirms dependencies:

- checks Postgres connectivity
- checks Redis connectivity (if enabled)
- confirms the model manager is initialized

Example:

    curl http://localhost:8000/readyz

### GET `/metrics` — *Stable*

Prometheus-formatted metrics emitted by the API process.

Typical metrics:

- HTTP request counts and latencies
- cache hit/miss counters
- token generation counts
- rate-limit / quota events
- error breakdown by status code

Prometheus scrapes this endpoint when you run the full Docker stack (via `make up`). Grafana dashboards are pre-wired to the same Prometheus instance.

---

## 5. Database-Level Operations (via Makefile)

These operations are not HTTP endpoints; they are shell commands used by operators.

### Migrations (host / local mode)

Use the `ENV_FILE` mechanism to choose `.env` or `.env.local`, then:

    ENV_FILE=.env.local make migrate

This runs:

- Alembic `upgrade head` using the configured `DATABASE_URL`.

### Migrations (inside Docker API container)

When running in full container mode:

    make migrate-docker

This executes Alembic migrations in the `api` container, against the Postgres service from `docker-compose.yml`.

### Inspecting the DB with pgAdmin

With the stack running:

- Open `http://localhost:8080/pgadmin/` (through Nginx) or `http://localhost:5050` (direct bind) depending on your chosen topology.
- Log in using credentials from your `.env` file:
  - `PGADMIN_DEFAULT_EMAIL`
  - `PGADMIN_DEFAULT_PASSWORD`

From there, you can inspect:

- tables such as `api_keys`, `roles`, `completion_cache`, `inference_logs`
- schema migrations
- performance via Postgres extensions, if configured

---

## 6. Typical Operator Workflow

A common operations flow for a single-node deployment looks like:

1. **Start the stack**

       make up

2. **Seed an admin key** (once per environment)

       make seed-key API_KEY=<admin-key>

3. **Verify health**

       curl http://localhost:8000/healthz
       curl http://localhost:8000/readyz

4. **Monitor metrics**

       # Prometheus scrape target
       http://localhost:8080/api/metrics

   Then open Grafana via:

       http://localhost:8080/grafana/

5. **Check usage for a given key**

       curl http://localhost:8080/api/v1/me/usage \
         -H "X-API-Key: <user-key>"

6. **Inspect logs (when debugging)**

       curl http://localhost:8080/api/v1/admin/logs \
         -H "X-API-Key: <admin-key>"

7. **Manage keys**

   - list: `GET /v1/admin/keys`
   - create: `POST /v1/admin/keys`
   - update: `PATCH /v1/admin/keys/{key}`
   - revoke: `DELETE /v1/admin/keys/{key}`

Together, these capabilities give you a small but realistic operational surface similar to what you would find in production-style LLM infrastructure: authentication, usage tracking, observability, and controlled administration.