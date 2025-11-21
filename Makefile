# ====== Config ======
ENV_FILE ?= .env

# helper to load .env vars in recipes
define dotenv
	set -a; \
	[ -f $(ENV_FILE) ] && . $(ENV_FILE); \
	set +a;
endef

API_PORT      ?= 8000          # FastAPI app (host)
LLM_PORT      ?= 9001          # LLM runtime (host)
NGINX_PORT    ?= 8080          # public entry
PG_HOST       ?= 127.0.0.1
PG_PORT       ?= 5433
PG_USER       ?= llm
PG_DB         ?= llm
REDIS_HOST    ?= 127.0.0.1
REDIS_PORT    ?= 6379

# You can pass API_KEY on the command line: make seed-key API_KEY=sk_live_...
API_KEY       ?=

# ====== Phony targets ======
.PHONY: dev dev-tmux up down restart ps status logs logs-nginx logs-postgres logs-redis \
migrate revision seed-key curl test api llm env clean clean-volumes nuke

# ====== One-shot developer experience ======
dev: up migrate ## Bring up infra, run migrations. Run `make api` and `make llm` in two terminals.
	@echo "✅ Infra is up and DB is migrated."
	@echo "👉 In a separate terminal: make api"
	@echo "👉 In another terminal:  make llm"
	@echo "👉 Then test:           make curl"

# Optional: auto-run API + LLM in tmux panes
dev-tmux: up migrate
	@command -v tmux >/dev/null || (echo "tmux not found. Install it or use 'make dev'." && exit 1)
	tmux new-session -d -s llmdev "make api"
	tmux split-window -h "make llm"
	tmux split-window -v "make logs-nginx"
	tmux select-pane -t 0
	tmux attach-session -t llmdev

# ====== Containers ======
up: ## Start docker services (Postgres, Redis, Prometheus, Grafana, pgAdmin, Nginx)
	docker compose up -d postgres redis prometheus grafana pgadmin nginx
	@echo "⏳ Waiting for Postgres @ $(PG_HOST):$(PG_PORT) ..."
	@for i in $$(seq 1 30); do \
		pg_isready -h $(PG_HOST) -p $(PG_PORT) -d $(PG_DB) -U $(PG_USER) >/dev/null 2>&1 && break; \
		sleep 1; \
	done || (echo "❌ Postgres not ready on $(PG_HOST):$(PG_PORT)"; exit 1)
	@echo "✅ Docker services are up."

down: ## Stop docker services (keep volumes)
	docker compose down

restart: down up ## Restart services

ps: ## List compose services
	docker compose ps

status: ps ## Alias

logs: ## Tail all docker logs
	docker compose logs -f

logs-nginx:
	docker compose logs -f nginx

logs-postgres:
	docker compose logs -f postgres

logs-redis:
	docker compose logs -f redis

# ====== DB Migrations ======
migrate: ## Alembic upgrade to head (uses your .env DATABASE_URL)
	uv run alembic upgrade head

revision: ## Autogenerate a new migration (edit message: make revision m="msg")
	@if [ -z "$(m)" ]; then echo 'Usage: make revision m="your message"'; exit 1; fi
	uv run alembic revision --autogenerate -m "$(m)"

# ====== Seed admin API key ======
seed-key: ## Insert 'admin' role + API key into Postgres (requires API_KEY=...)
	@if [ -z "$(API_KEY)" ]; then echo '❌ Provide API_KEY, e.g. make seed-key API_KEY=$$(openssl rand -hex 24)'; exit 1; fi
	docker exec -i llm_postgres psql -U $(PG_USER) -d $(PG_DB) -v ON_ERROR_STOP=1 \
	  -c "DO $$$$ BEGIN IF NOT EXISTS (SELECT 1 FROM roles WHERE name='admin') THEN INSERT INTO roles (name) VALUES ('admin'); END IF; END $$$$;"
	docker exec -i llm_postgres psql -U $(PG_USER) -d $(PG_DB) -v ON_ERROR_STOP=1 \
	  -c "INSERT INTO api_keys (key, label, active, role_id, quota_used, quota_monthly, quota_reset_at) SELECT '$(API_KEY)', 'bootstrap', TRUE, r.id, 0, NULL, NULL FROM roles r WHERE r.name = 'admin' ON CONFLICT (key) DO NOTHING;"
	@echo "✅ Seeded API key: $(API_KEY)"

# ====== Local runners (host) ======
api: ## Run the App API (FastAPI) on $(API_PORT)
	ENV=dev PORT=$(API_PORT) uv run serve

llm: ## Run the LLM runtime on $(LLM_PORT)
	uv run uvicorn app.services.llm_api:app --host 0.0.0.0 --port $(LLM_PORT)

# ====== Quick checks ======
curl: ## Example request via Nginx (requires API_KEY to be seeded)
	@if [ -z "$(API_KEY)" ]; then echo 'Tip: make curl API_KEY=<your-key>'; fi
	curl -s http://localhost:$(NGINX_PORT)/api/v1/generate \
	  -H "Content-Type: application/json" \
	  -H "X-API-Key: $(API_KEY)" \
	  -d '{ "prompt": "Write a haiku about autumn leaves.", "max_new_tokens": 32, "temperature": 0.7, "top_p": 0.95 }' | jq .

test: ## Sanity: hit LLM directly; then hit API via Nginx (needs API_KEY)
	curl -s http://127.0.0.1:$(LLM_PORT)/generate \
	  -H "Content-Type: application/json" \
	  -d '{"prompt":"ping","max_new_tokens":4}' | jq .
	@if [ -n "$(API_KEY)" ]; then \
	  curl -s http://localhost:$(NGINX_PORT)/api/v1/generate \
	    -H "Content-Type: application/json" \
	    -H "X-API-Key: $(API_KEY)" \
	    -d '{ "prompt": "ping", "max_new_tokens": 4 }' | jq . ; \
	else echo "⚠️  Set API_KEY to test through Nginx: make test API_KEY=<key>"; fi

env: ## Print key env vars
	@echo "DATABASE_URL=$(DATABASE_URL)"
	@echo "REDIS_URL=$(REDIS_URL)"
	@echo "LLM_SERVICE_URL=$(LLM_SERVICE_URL)"
	@echo "API_PORT=$(API_PORT)  LLM_PORT=$(LLM_PORT)  NGINX_PORT=$(NGINX_PORT)"

# ====== Cleanup (careful!) ======
clean: ## Stop containers & remove orphans (keep volumes)
	docker compose down --remove-orphans

clean-volumes: ## Stop containers & REMOVE VOLUMES (DB/Redis data LOST)
	@read -p "This will DELETE volumes (DB/Redis). Are you sure? [y/N] " ans; \
	if [ "$$ans" = "y" ] || [ "$$ans" = "Y" ]; then \
	  docker compose down -v; \
	else \
	  echo "Aborted."; \
	fi

nuke: ## Remove containers, images, volumes of this project (extreme)
	@read -p "This will nuke containers/images/volumes for this project. Continue? [y/N] " ans; \
	if [ "$$ans" = "y" ] || [ "$$ans" = "Y" ]; then \
	  docker compose down -v --rmi local --remove-orphans; \
	else \
	  echo "Aborted."; \
	fi