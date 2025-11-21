# app/main.py
from __future__ import annotations

import orjson
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core import logging as logging_config
from app.core import metrics, limits
from app.core.redis import init_redis, close_redis   # <-- add this import


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.service_name,
        description="Backend service for running LLM inference",
        version=settings.version,
        debug=settings.debug,
        json_dumps=lambda v, *, default: orjson.dumps(v, default=default).decode(),
        json_loads=orjson.loads,
    )

    # --- Middleware: CORS, access logging, concurrency guard ---
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    logging_config.setup(app)
    limits.setup(app)

    # --- Metrics (/metrics) ---
    metrics.setup(app)

    # --- Routers ---
    from app.api import health, generate
    app.include_router(health.router)
    app.include_router(generate.router, prefix="/v1")

    # --- Startup: log config (existing) ---
    @app.on_event("startup")
    async def _log_startup():
        import logging
        logging.getLogger("uvicorn.error").info(
            "CORS allow_origins=%s | env=%s | debug=%s | redis_enabled=%s",
            settings.cors_allowed_origins,
            settings.env,
            settings.debug,
            settings.redis_enabled,
        )

    # --- Startup: init Redis (new) ---
    @app.on_event("startup")
    async def _init_redis():
        try:
            app.state.redis = await init_redis()
        except Exception as e:
            # Don’t crash the app if Redis is optional in your env
            import logging
            logging.getLogger("uvicorn.error").exception("Redis init failed: %s", e)
            app.state.redis = None

    # --- Shutdown: close Redis (new) ---
    @app.on_event("shutdown")
    async def _shutdown_redis():
        await close_redis(getattr(app.state, "redis", None))

    return app


# For uvicorn without --factory; harmless when using factory too
app = create_app()