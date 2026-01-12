# Makefile for llm-server project
# ====== Config ======
ENV_FILE ?= .env
PROJECT_NAME ?= llm-server

define dotenv
	set -a; \
	[ -f $(ENV_FILE) ] && . $(ENV_FILE); \
	set +a;
endef

API_PORT   ?= 8000
NGINX_PORT ?= 8080

PG_HOST ?= 127.0.0.1
PG_PORT ?= 5433
PG_USER ?= llm
PG_DB   ?= llm

REDIS_HOST ?= 127.0.0.1
REDIS_PORT ?= 6379

API_KEY ?=

# ====== Compose commands (DELTAS) ======
COMPOSE_BASE  := COMPOSE_PROJECT_NAME=$(PROJECT_NAME) docker compose
COMPOSE_PROD  := $(COMPOSE_BASE) -f docker-compose.yml
COMPOSE_DEV   := $(COMPOSE_BASE) -f docker-compose.yml -f docker-compose.dev.yml
COMPOSE_LOCAL := $(COMPOSE_BASE) -f docker-compose.yml -f docker-compose.local.yml

.PHONY: \
  init init-env bootstrap \
  dev-local dev-cpu dev-tmux \
  up up-cpu up-local up-prod \
  down down-local down-prod restart ps status \
  logs logs-nginx logs-postgres logs-redis \
  config config-dev config-local config-prod \
  migrate revision migrate-docker seed-key seed-key-from-env \
  api-local curl test env \
  clean clean-volumes nuke

# ====== Setup ======
init: init-env

init-env:
	@if [ -f .env ]; then \
		echo "✔ .env already exists (using $(ENV_FILE))"; \
	elif [ -f .env.example ]; then \
		echo "📄 Creating .env from .env.example"; \
		cp .env.example .env; \
	else \
		echo "❌ .env.example not found; create .env manually."; \
		exit 1; \
	fi
	@echo "✅ Env file ready: .env"

bootstrap: init up seed-key-from-env

# ====== Sanity checks ======
config: config-dev
config-dev:
	@$(COMPOSE_DEV) config >/dev/null && echo "✅ compose config (DEV) OK"
config-local:
	@$(COMPOSE_LOCAL) config >/dev/null && echo "✅ compose config (LOCAL) OK"
config-prod:
	@$(COMPOSE_PROD) config >/dev/null && echo "✅ compose config (PROD) OK"

# ====== Golden paths ======
dev-cpu: ENV_FILE=.env
dev-cpu: up-cpu migrate-docker
	@echo "✅ Dev container stack is up and DB migrated."
	@echo "👉 Seed key: make seed-key-from-env"
	@echo "👉 Test: make curl API_KEY=<your-key>"

dev-local: ENV_FILE=.env.local
dev-local: up-local
	@echo "✅ Local infra is up."
	@echo "👉 Run API on host: ENV_FILE=.env.local make api-local"
	@echo "👉 Migrate on host:  ENV_FILE=.env.local make migrate"

dev-tmux: ENV_FILE=.env.local
dev-tmux: up-local migrate
	@command -v tmux >/dev/null || (echo "tmux not found. Install it or use 'make dev-local'." && exit 1)
	tmux new-session -d -s llmdev "make api-local"
	tmux split-window -h "make logs-nginx"
	tmux select-pane -t 0
	tmux attach-session -t llmdev

# ====== Compose up/down ======
up: up-cpu ## default: DEV container stack
up-cpu:
	@if [ ! -f $(ENV_FILE) ]; then \
		echo "❌ $(ENV_FILE) not found. Run 'make init' first."; \
		exit 1; \
	fi
	$(COMPOSE_DEV) up -d --build
	@echo "⏳ Waiting for Postgres @ $(PG_HOST):$(PG_PORT) ..."
	@for i in $$(seq 1 30); do \
		pg_isready -h $(PG_HOST) -p $(PG_PORT) -d $(PG_DB) -U $(PG_USER) >/dev/null 2>&1 && break; \
		sleep 1; \
	done || (echo "❌ Postgres not ready on $(PG_HOST):$(PG_PORT)"; exit 1)
	@echo "✅ DEV container stack is up."

up-local:
	$(COMPOSE_LOCAL) up -d --build --scale api=0
	@echo "⏳ Waiting for Postgres @ $(PG_HOST):$(PG_PORT) ..."
	@for i in $$(seq 1 30); do \
		pg_isready -h $(PG_HOST) -p $(PG_PORT) -d $(PG_DB) -U $(PG_USER) >/dev/null 2>&1 && break; \
		sleep 1; \
	done || (echo "❌ Postgres not ready on $(PG_HOST):$(PG_PORT)"; exit 1)
	@echo "✅ LOCAL infra stack is up."

up-prod:
	@if [ ! -f $(ENV_FILE) ]; then \
		echo "❌ $(ENV_FILE) not found. Run 'make init' first."; \
		exit 1; \
	fi
	$(COMPOSE_PROD) up -d --build
	@echo "✅ PROD stack is up."

down:
	$(COMPOSE_DEV) down --remove-orphans

down-local:
	$(COMPOSE_LOCAL) down --remove-orphans

down-prod:
	$(COMPOSE_PROD) down --remove-orphans

restart: down up

ps:
	$(COMPOSE_DEV) ps

status: ps

logs:
	$(COMPOSE_DEV) logs -f --tail=200

logs-nginx:
	$(COMPOSE_DEV) logs -f --tail=200 nginx

logs-postgres:
	$(COMPOSE_DEV) logs -f --tail=200 postgres

logs-redis:
	$(COMPOSE_DEV) logs -f --tail=200 redis

# ====== DB Migrations ======
migrate:
	@$(dotenv) \
	uv run python -m alembic upgrade head

revision:
	@if [ -z "$(m)" ]; then echo 'Usage: make revision m="your message"'; exit 1; fi
	@$(dotenv) \
	uv run python -m alembic revision --autogenerate -m "$(m)"

migrate-docker:
	$(COMPOSE_DEV) exec api python -m alembic upgrade head

# ====== Seed API key ======
seed-key:
	@if [ -z "$(API_KEY)" ]; then echo '❌ Provide API_KEY, e.g. make seed-key API_KEY=$$(openssl rand -hex 24)'; exit 1; fi
	docker exec -i llm_postgres psql -U $(PG_USER) -d $(PG_DB) -v ON_ERROR_STOP=1 \
	  -c "INSERT INTO roles (name) SELECT 'admin' WHERE NOT EXISTS (SELECT 1 FROM roles WHERE name = 'admin');"
	docker exec -i llm_postgres psql -U $(PG_USER) -d $(PG_DB) -v ON_ERROR_STOP=1 \
	  -c "INSERT INTO api_keys (key, name, label, active, role_id, quota_used, quota_monthly, quota_reset_at) \
	      SELECT '$(API_KEY)', 'bootstrap', 'bootstrap', TRUE, r.id, 0, NULL, NULL \
	      FROM roles r WHERE r.name = 'admin' \
	      ON CONFLICT (key) DO NOTHING;"
	@echo "✅ Seeded API key: $(API_KEY)"

seed-key-from-env:
	@$(dotenv); \
	if [ -z "$$API_KEY" ]; then \
		echo "❌ API_KEY not set in $(ENV_FILE)."; \
		exit 1; \
	fi; \
	API_KEY="$$API_KEY" $(MAKE) seed-key

# ====== Local API runner ======
api-local:
	@$(dotenv) \
	ENV=dev PORT=$(API_PORT) uv run serve

# ====== Quick checks ======
curl:
	@if [ -z "$(API_KEY)" ]; then echo 'Tip: make curl API_KEY=<your-key>'; fi
	@url="http://localhost:$(NGINX_PORT)/api/v1/generate"; \
	echo "➡️  Requesting $$url"; \
	curl -sS --fail "$$url" \
	  -H "Content-Type: application/json" \
	  -H "X-API-Key: $(API_KEY)" \
	  -d '{ "prompt": "Write a haiku about autumn leaves.", "max_new_tokens": 32, "temperature": 0.7, "top_p": 0.95 }' \
	  | jq .

env:
	@$(dotenv) \
	echo "ENV_FILE=$(ENV_FILE)"; \
	echo "DATABASE_URL=$$DATABASE_URL"; \
	echo "REDIS_URL=$$REDIS_URL"; \
	echo "PROJECT_NAME=$(PROJECT_NAME)"

# ====== Cleanup ======
clean:
	$(COMPOSE_DEV) down --remove-orphans

clean-volumes:
	@read -p "This will DELETE volumes (DB/Redis). Are you sure? [y/N] " ans; \
	if [ "$$ans" = "y" ] || [ "$$ans" = "Y" ]; then \
	  $(COMPOSE_DEV) down -v --remove-orphans; \
	else \
	  echo "Aborted."; \
	fi

nuke:
	@echo "🔨 Hard nuke for Docker resources with project '$(PROJECT_NAME)'"
	@ids=$$(docker ps -aq --filter "label=com.docker.compose.project=$(PROJECT_NAME)"); \
	if [ -n "$$ids" ]; then docker rm -f $$ids >/dev/null 2>&1 || true; fi
	@net_ids=$$(docker network ls -q --filter "label=com.docker.compose.project=$(PROJECT_NAME)"); \
	if [ -n "$$net_ids" ]; then docker network rm $$net_ids >/dev/null 2>&1 || true; fi
	@read -p "❗ Delete volumes for project '$(PROJECT_NAME)' as well? [y/N] " ans; \
	if [ "$$ans" = "y" ] || [ "$$ans" = "Y" ]; then \
		vol_ids=$$(docker volume ls -q --filter "label=com.docker.compose.project=$(PROJECT_NAME)"); \
		if [ -n "$$vol_ids" ]; then docker volume rm $$vol_ids >/dev/null 2>&1 || true; fi; \
	fi
	@echo "✅ Hard nuke complete."