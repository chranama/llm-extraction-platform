# src/llm_server/main.py
from __future__ import annotations

import os
import orjson
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from llm_server.core.config import get_settings
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
    s = get_settings()
    default = "eager" if s.env.strip().lower() == "prod" else "lazy"
    return os.getenv("MODEL_LOAD_MODE", default).strip().lower()


def _model_warmup_enabled(mode: str) -> bool:
    """
    Optional extra safety: after ensure_loaded(), run a tiny generation to confirm
    the model can actually execute on the configured device.

    Defaults:
      - prod + eager/on => enabled
      - otherwise => disabled
    Override with MODEL_WARMUP=0/1.
    """
    raw = os.getenv("MODEL_WARMUP")
    if raw is not None:
        return raw.strip().lower() in ("1", "true", "yes", "y", "on")

    s = get_settings()
    is_prod = s.env.strip().lower() == "prod"
    return is_prod and mode in ("eager", "on")


def _warmup_prompt() -> str:
    return os.getenv("MODEL_WARMUP_PROMPT", "Say 'ok'.")


def _warmup_max_new_tokens() -> int:
    try:
        return int(os.getenv("MODEL_WARMUP_MAX_NEW_TOKENS", "8"))
    except Exception:
        return 8


@asynccontextmanager
async def lifespan(app: FastAPI):
    import logging

    s = get_settings()
    mode = _model_load_mode()

    logging.getLogger("uvicorn.error").info(
        "CORS allow_origins=%s | env=%s | debug=%s | redis_enabled=%s | model_load_mode=%s",
        s.cors_allowed_origins,
        s.env,
        s.debug,
        s.redis_enabled,
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

            # "eager"/"on": load at startup
            if mode in ("eager", "on"):
                if not hasattr(llm, "ensure_loaded"):
                    app.state.model_loaded = False
                    app.state.model_error = "LLM backend has no ensure_loaded(); cannot eager load"
                else:
                    llm.ensure_loaded()
                    app.state.model_loaded = True

                    # Optional warmup smoke test
                    if _model_warmup_enabled(mode):
                        try:
                            prompt = _warmup_prompt()
                            max_new = _warmup_max_new_tokens()
                            out = llm.generate(prompt=prompt, max_new_tokens=max_new, temperature=0.0)
                            _ = out
                        except Exception as e:
                            app.state.model_loaded = False
                            app.state.model_error = f"warmup_failed: {repr(e)}"
                            logging.getLogger("uvicorn.error").exception("Model warmup failed: %s", e)
                            logging.getLogger("uvicorn.error").error("Aborting startup: warmup failed in eager mode")
                            raise

            # "lazy": initialized but not loaded
            else:
                app.state.model_loaded = False

        except Exception as e:
            app.state.model_error = repr(e)

            if mode in ("eager", "on"):
                logging.getLogger("uvicorn.error").exception("LLM eager init failed; aborting startup: %s", e)
                raise

            logging.getLogger("uvicorn.error").exception("LLM init failed (lazy mode continues): %s", e)
            app.state.llm = None
            app.state.model_loaded = False

    yield

    # --------------------
    # Shutdown
    # --------------------
    await close_redis(getattr(app.state, "redis", None))


def create_app() -> FastAPI:
    s = get_settings()

    app = FastAPI(
        title=s.service_name,
        description="Backend service for running LLM inference",
        version=s.version,
        debug=s.debug,
        lifespan=lifespan,
        json_dumps=lambda v, *, default: orjson.dumps(v, default=default).decode(),
        json_loads=orjson.loads,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=s.cors_allowed_origins,
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