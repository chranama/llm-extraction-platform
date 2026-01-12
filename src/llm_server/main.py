#  src/llm_server/main.py
from __future__ import annotations

import os
import orjson
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from llm_server.core.config import settings
from llm_server.core import logging as logging_config
from llm_server.core import metrics, limits
from llm_server.core import errors
from llm_server.core.redis import init_redis, close_redis
from llm_server.services.llm import build_llm_from_settings


def _model_load_mode() -> str:
    """
    Values:
      - "off"   : never build/load LLM in lifespan
      - "lazy"  : build LLM object, but don't ensure_loaded at startup
      - "eager" : build + ensure_loaded at startup
      - "on"    : alias for "eager"
    """
    default = "eager" if settings.env.strip().lower() == "prod" else "lazy"
    return os.getenv("MODEL_LOAD_MODE", default).strip().lower()


@asynccontextmanager
async def lifespan(app: FastAPI):
    import logging

    mode = _model_load_mode()

    logging.getLogger("uvicorn.error").info(
        "CORS allow_origins=%s | env=%s | debug=%s | redis_enabled=%s | model_load_mode=%s",
        settings.cors_allowed_origins,
        settings.env,
        settings.debug,
        settings.redis_enabled,
        mode,
    )

    # --------------------
    # Redis startup
    # --------------------
    try:
        app.state.redis = await init_redis()
    except Exception as e:
        logging.getLogger("uvicorn.error").exception("Redis init failed: %s", e)
        app.state.redis = None

    # --------------------
    # LLM startup
    # --------------------
    app.state.llm = None
    app.state.model_load_mode = mode
    app.state.model_loaded = False
    app.state.model_error = None

    if mode != "off":
        try:
            llm = build_llm_from_settings()
            app.state.llm = llm

            if mode in ("eager", "on"):
                if hasattr(llm, "ensure_loaded"):
                    llm.ensure_loaded()
                    app.state.model_loaded = True
                else:
                    # Can't force a load for this backend; treat as not loaded
                    app.state.model_loaded = False
            else:
                # lazy: not loaded yet, but initialized
                app.state.model_loaded = False

        except Exception as e:
            app.state.model_error = repr(e)

            # If you want true "hard eager", crash startup when eager fails.
            # This prevents the service from looking "up" while it can't ever serve.
            if mode in ("eager", "on"):
                logging.getLogger("uvicorn.error").exception("LLM eager init failed; aborting startup: %s", e)
                raise

            # In lazy mode, you can choose to keep API up and report via /modelz
            logging.getLogger("uvicorn.error").exception("LLM init failed (lazy mode continues): %s", e)
            app.state.llm = None

    yield

    # --------------------
    # Shutdown
    # --------------------
    await close_redis(getattr(app.state, "redis", None))


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.service_name,
        description="Backend service for running LLM inference",
        version=settings.version,
        debug=settings.debug,
        lifespan=lifespan,
        json_dumps=lambda v, *, default: orjson.dumps(v, default=default).decode(),
        json_loads=orjson.loads,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    logging_config.setup(app)
    limits.setup(app)
    metrics.setup(app)
    errors.setup(app)

    from llm_server.api import health, generate, models, admin, extract

    app.include_router(health.router)
    app.include_router(generate.router)
    app.include_router(models.router)
    app.include_router(admin.router)
    app.include_router(extract.router)

    return app


app = create_app()