> [!WARNING]
> Historical snapshot; may not reflect current implementation.

# Project Structure

The repository is organized into clear functional layers: API gateway, model orchestration, evaluation tools, observability stack, UI playground, and deployment configuration.  

# Directory Highlights

### **src/llm_server/**
Core application code:
- **api/** — all FastAPI endpoints (generate, health, admin, models)
- **core/** — configuration, errors, rate limits, logging, metrics, Redis helpers
- **db/** — SQLAlchemy models + async session
- **services/** — LLM execution layer, multimodel routing, API orchestration
- **providers/** — HTTP client wrappers for remote models
- **eval/** — evaluation framework (GSM8K, MMLU, MBPP, etc.)

### **ui/**
Self-contained Vite/React frontend for the LLM Playground.

### **infra/**
Infrastructure layer:
- Prometheus configs  
- Grafana dashboards  
- Nginx reverse proxy configs  

### **migrations/**
Alembic migration files and environment.

### **scripts/**
Utility scripts:
- seed keys  
- run local/server  
- download HF models  
- migrate DB  

### **tests/**
Comprehensive pytest suite covering:
auth, quotas, generate, models, metrics, rate limits, integration flow.

Below is the actual directory structure (truncated for readability but accurate to the project).

llm-server/
├── alembic.ini
├── data/
│   ├── app.db
│   └── test_app.db
├── docker-compose.yml
├── docker-compose.local.yml
├── docker-compose.dev.yml
├── Dockerfile.api
├── docs/
│   ├── 00-intro.md
│   ├── 01-architecture.md
│   ├── 02-features.md
│   ├── 03-observability.md
│   ├── 04-caching.md
│   ├── 05-multimodel.md
│   ├── 06-api-versioning.md
│   ├── 07-project-structure.md
│   ├── 08-makefile.md
│   ├── 09-quickstart-container.md
│   ├── 10-quickstart-local.md
│   ├── 11-admin-ops.md
│   └── 12-testing.md
├── infra/
│   ├── grafana/
│   │   ├── dashboards/
│   │   │   ├── llm-api-overview.json
│   │   │   └── prometheus-full.json
│   │   └── provisioning/
│   │       ├── dashboards/dashboards.yml
│   │       └── datasources/datasources.yml
│   ├── nginx.conf
│   ├── nginx.local.conf
│   ├── prometheus.yml
│   └── prometheus.local.yml
├── llm_server.egg-info/
├── Makefile
├── migrations/
│   ├── env.py
│   ├── README
│   ├── script.py.mako
│   └── versions/
│       ├── 32dab146fc14_baseline_schema.py
│       └── 4b11bcee269e_create_core_tables.py
├── models.yaml
├── pyproject.toml
├── README.md
├── scripts/
│   ├── download-hf-model.sh
│   ├── list_api_keys.py
│   ├── migrate_data.py
│   ├── run-cpu.sh
│   ├── run-local.sh
│   ├── run-llm-api-local.sh
│   └── seed_api_key.py
├── src/
│   └── llm_server/
│       ├── api/
│       │   ├── admin.py
│       │   ├── deps.py
│       │   ├── generate.py
│       │   ├── health.py
│       │   └── models.py
│       ├── cli.py
│       ├── core/
│       │   ├── config.py
│       │   ├── errors.py
│       │   ├── limits.py
│       │   ├── logging.py
│       │   ├── metrics.py
│       │   └── redis.py
│       ├── db/
│       │   ├── models.py
│       │   └── session.py
│       ├── eval/
│       │   ├── cli.py
│       │   ├── client/http_client.py
│       │   ├── datasets/
│       │   │   ├── gsm8k.py
│       │   │   ├── mbpp.py
│       │   │   ├── mmlu.py
│       │   │   ├── summarization.py
│       │   │   └── toxicity.py
│       │   ├── metrics/exact_match.py
│       │   ├── prompts/gsm8k.py
│       │   ├── reporting/
│       │   ├── results/
│       │   └── runners/
│       │       ├── base.py
│       │       ├── gsm8k_runner.py
│       │       ├── mbpp_runner.py
│       │       ├── mmlu_runner.py
│       │       ├── summarization_runner.py
│       │       └── toxicity_runner.py
│       ├── main.py
│       ├── providers/llm_client.py
│       └── services/
│           ├── llm_api.py
│           ├── llm.py
│           └── multimodel.py
├── tests/
│   ├── conftest.py
│   ├── test_auth.py
│   ├── test_auth_rate_quota.py
│   ├── test_generate.py
│   ├── test_health.py
│   ├── test_health_metrics.py
│   ├── test_integrate_generate.py
│   └── test_limits.py
├── ui/
│   ├── Dockerfile.ui
│   ├── index.html
│   ├── package.json
│   ├── src/
│   │   ├── App.tsx
│   │   ├── components/Playground.tsx
│   │   ├── lib/api.ts
│   │   └── main.tsx
│   ├── tsconfig.json
│   ├── tsconfig.app.json
│   └── vite.config.ts
└── uv.lock

