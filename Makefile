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
PG_PASSWORD ?= llm
PG_DB   ?= llm

REDIS_HOST ?= 127.0.0.1
REDIS_PORT ?= 6379

API_KEY ?=

# ====== Paths ======
COMPOSE_DIR ?= deploy/compose
COMPOSE_YML ?= $(COMPOSE_DIR)/docker-compose.yml
COMPOSE_DEV_YML ?= $(COMPOSE_DIR)/docker-compose.dev.yml
COMPOSE_LOCAL_YML ?= $(COMPOSE_DIR)/docker-compose.local.yml
COMPOSE_EVAL_YML ?= $(COMPOSE_DIR)/docker-compose.eval.yml

BACKEND_DIR ?= backend
EVAL_ARGS ?=

# ====== Compose commands (DELTAS) ======
COMPOSE_BASE  := COMPOSE_PROJECT_NAME=$(PROJECT_NAME) docker compose --env-file $(ENV_FILE)
COMPOSE_PROD  := $(COMPOSE_BASE) -f $(COMPOSE_YML)
COMPOSE_DEV   := $(COMPOSE_BASE) -f $(COMPOSE_YML) -f $(COMPOSE_DEV_YML)
COMPOSE_LOCAL := $(COMPOSE_BASE) -f $(COMPOSE_YML) -f $(COMPOSE_LOCAL_YML)
COMPOSE_EVAL  := $(COMPOSE_BASE) -f $(COMPOSE_YML) -f $(COMPOSE_EVAL_YML)

.PHONY: \
  init init-env bootstrap \
  dev-local dev-cpu dev-tmux \
  up up-cpu up-local up-prod \
  down down-local down-prod restart ps status \
  logs logs-nginx logs-postgres logs-redis \
  config config-dev config-local config-prod \
  migrate revision migrate-docker seed-key seed-key-from-env \
  api-local curl test env \
  test-unit test-integration test-all test-integration-localdb \
  clean clean-volumes nuke \
  eval-build eval-help eval-run eval-shell eval-up eval-down \
  int-up int-down int-ps int-logs int-test int-test-api int-test-nginx

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
	@echo "👉 Backend tests: make test-all (or test-integration-localdb)"

dev-local: ENV_FILE=.env.local
dev-local: up-local
	@echo "✅ Local infra is up."
	@echo "👉 Run API on host: ENV_FILE=.env.local make api-local"
	@echo "👉 Migrate on host:  ENV_FILE=.env.local make migrate"
	@echo "👉 Backend tests:   ENV_FILE=.env.local make test-all"

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
	$(COMPOSE_LOCAL) up -d --scale api=0
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

# ====== DB Migrations (host) ======
migrate:
	@$(dotenv) \
	cd $(BACKEND_DIR) && \
	APP_ROOT=.. uv run python -m alembic upgrade head

revision:
	@if [ -z "$(m)" ]; then echo 'Usage: make revision m="your message"'; exit 1; fi
	@$(dotenv) \
	cd $(BACKEND_DIR) && \
	APP_ROOT=.. uv run python -m alembic revision --autogenerate -m "$(m)"

migrate-docker:
	$(COMPOSE_DEV) exec -T api python -m alembic upgrade head

# ====== Seed API key ======
seed-key:
	@if [ -z "$(API_KEY)" ]; then echo '❌ Provide API_KEY, e.g. make seed-key API_KEY=$$(openssl rand -hex 24)'; exit 1; fi
	$(COMPOSE_DEV) exec -T postgres psql -U $(PG_USER) -d $(PG_DB) -v ON_ERROR_STOP=1 \
	  -c "INSERT INTO roles (name) SELECT 'admin' WHERE NOT EXISTS (SELECT 1 FROM roles WHERE name = 'admin');"
	$(COMPOSE_DEV) exec -T postgres psql -U $(PG_USER) -d $(PG_DB) -v ON_ERROR_STOP=1 \
	  -c "INSERT INTO api_keys (key, name, label, active, role_id, quota_used, quota_monthly, quota_reset_at) \
	      SELECT '$(API_KEY)', 'bootstrap', 'bootstrap', TRUE, r.id, 0, NULL, NULL \
	      FROM roles r WHERE r.name = 'admin' \
	      ON CONFLICT (key) DO NOTHING;"
	@echo "✅ Seeded API key: $(API_KEY)"

seed-key-from-env:
	@$(dotenv) \
	if [ -z "$$API_KEY" ]; then \
		echo "❌ API_KEY not set in $(ENV_FILE)."; \
		exit 1; \
	fi; \
	API_KEY="$$API_KEY" $(MAKE) seed-key

# ====== Local API runner ======
api-local:
	@$(dotenv) \
	cd $(BACKEND_DIR) && \
	APP_ROOT=.. PORT=$(API_PORT) uv run serve

# ====== Backend tests (the new smoke test path) ======
# NOTE: backend integration tests require a host-reachable DATABASE_URL.
# .env (docker network):  postgresql+asyncpg://...@postgres:5432/...
# .env.local (host):      postgresql+asyncpg://...@127.0.0.1:5433/...
test-unit:
	@$(dotenv) \
	cd $(BACKEND_DIR) && \
	PYTHONPATH=src uv run pytest -q -m unit

test-integration:
	@$(dotenv) \
	cd $(BACKEND_DIR) && \
	PYTHONPATH=src uv run pytest -q -m integration

test-all:
	@$(dotenv) \
	cd $(BACKEND_DIR) && \
	PYTHONPATH=src uv run pytest -q

# Convenience: ensure DATABASE_URL points at host-mapped compose Postgres
test-integration-localdb: ENV_FILE=.env.local
test-integration-localdb: up-local
	@$(dotenv); \
	if [ -z "$$DATABASE_URL" ]; then \
	  export DATABASE_URL="postgresql+asyncpg://$(PG_USER):$(PG_PASSWORD)@$(PG_HOST):$(PG_PORT)/$(PG_DB)"; \
	fi; \
	cd $(BACKEND_DIR) && PYTHONPATH=src uv run pytest -q -m integration

# ====== Eval (containerized) ======
eval-up: ENV_FILE=.env
eval-up:
	@if [ ! -f $(ENV_FILE) ]; then \
		echo "❌ $(ENV_FILE) not found. Run 'make init' first."; \
		exit 1; \
	fi
	@$(COMPOSE_EVAL) --profile eval up -d --build api postgres redis
	@echo "✅ Eval dependencies up (api/postgres/redis)."

eval-down:
	@$(COMPOSE_EVAL) --profile eval down --remove-orphans
	@echo "✅ Eval stack down."

eval-build:
	@$(COMPOSE_EVAL) --profile eval build eval

eval-help:
	@$(COMPOSE_EVAL) --profile eval run --rm eval eval --help

eval-run: eval-up
	@$(COMPOSE_EVAL) --profile eval run --rm eval sh -lc "pip install -e /work/eval && eval $(EVAL_ARGS)"

eval-shell: eval-up
	@$(COMPOSE_EVAL) --profile eval run --rm --entrypoint sh eval

# ====== Integration tests (external harness; runs against running stack) ======
int-up: ENV_FILE=.env
int-up:
	@if [ ! -f $(ENV_FILE) ]; then \
		echo "❌ $(ENV_FILE) not found. Run 'make init' first."; \
		exit 1; \
	fi
	$(COMPOSE_DEV) up -d --build
	@echo "✅ Integration stack up (DEV)."

int-down:
	$(COMPOSE_DEV) down --remove-orphans

int-ps:
	$(COMPOSE_DEV) ps

int-logs:
	$(COMPOSE_DEV) logs -f --tail=200

int-test-nginx: ENV_FILE=.env
int-test-nginx: int-up
	@echo "🧪 Running integrations against nginx http://localhost:$(NGINX_PORT)/api"
	@$(dotenv) \
	INTEGRATION_BASE_URL="http://localhost:$(NGINX_PORT)/api" \
	uv run --project integrations pytest -q integrations/tests
	@echo "✅ Integration tests done."

int-test-api: ENV_FILE=.env
int-test-api: int-up
	@echo "🧪 Running integrations against api http://localhost:$(API_PORT)"
	@$(dotenv) \
	INTEGRATION_BASE_URL="http://localhost:$(API_PORT)" \
	uv run --project integrations pytest -q integrations/tests
	@echo "✅ Integration tests done."

int-test: int-test-nginx

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
	echo "PROJECT_NAME=$(PROJECT_NAME)"; \
	echo "COMPOSE_YML=$(COMPOSE_YML)"

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