# Quickstart: Containerized Deployment

This section describes the **simplest, most reliable path** to running the full LLM Server stack using Docker Compose.  
It is intended for users who want a production-style deployment without running the model locally.

The container stack includes:

- API Server (FastAPI + LLM runtime, CPU inference)
- Postgres (API keys, logs, cache)
- Redis (optional cache)
- Nginx (public entrypoint)
- Prometheus (metrics)
- Grafana (dashboards)
- pgAdmin (database UI)

---

## 1. Setup Environment

First create your environment file:

    make init

This ensures `.env` exists.  
You may edit it if desired (e.g., choose a different model or rate limits).

---

## 2. Start the Full Stack

Run:

    make up

This launches all containers and waits for Postgres to become ready.

Services become available at:

- **UI:**  
  http://localhost:8080/ui/

- **Public API (proxied through Nginx):**  
  http://localhost:8080/api/v1/generate

- **Direct FastAPI (not recommended for public exposure):**  
  http://localhost:8000/v1/generate

- **Prometheus:**  
  http://localhost:8080/prometheus/

- **Grafana:**  
  http://localhost:8080/grafana/  
  (Default login: `admin / admin`)

- **pgAdmin:**  
  http://localhost:8080/pgadmin/

---

## 3. Seed an Admin API Key

You must create an API key before making any requests:

    make seed-key API_KEY=$(openssl rand -hex 24)

This key becomes your authentication credential.  
You will pass it using the header:

    X-API-Key: <your-key>

---

## 4. Test the API (Optional)

Using the built-in helper:

    make curl API_KEY=<your-key>

or manually:

    curl -X POST http://localhost:8080/api/v1/generate \
      -H "Content-Type: application/json" \
      -H "X-API-Key: <your-key>" \
      -d '{ "prompt": "Say hello!", "max_new_tokens": 20 }'

A valid response looks like:

    {
      "model": "meta-llama/Llama-3.2-1B-Instruct",
      "output": "Hello there!",
      "cached": false
    }

---

## 5. View Logs

To view all service logs:

    make logs

Or individual services:

    make logs-nginx  
    make logs-postgres  
    make logs-redis

---

## 6. Stop or Reset

### Stop (keep data)

    make down

### Remove containers but keep data

    make clean

### Remove *everything* (containers, networks, volumes)

    make nuke  
(Use with caution—this fully resets the environment.)

---

## Summary

The minimal workflow for running the server is:

    make init
    make up
    make seed-key API_KEY=$(openssl rand -hex 24)
    make curl API_KEY=<your-key>

After this, your system is fully operational with:

- Authenticated LLM API
- Observability stack
- Admin interfaces
- Database persistence
