# server/src/llm_server/main.py
from __future__ import annotations

import os
import time
import orjson
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from llm_server.core.config import get_settings
from llm_server.core import logging as logging_config
from llm_server.core import metrics, limits
from llm_server.core import errors
from llm_server.core.redis import init_redis, close_redis
from llm_server.services.llm import build_llm_from_settings
from llm_server.io.policy_decisions import load_policy_decision_from_env


class RequestContextMiddleware(BaseHTTPMiddleware):
    """
    Sets minimal request context early so exception handlers can log failures with:
      - latency_ms (start_ts)
      - route/model_id/cached fallbacks

    IMPORTANT: must be added LAST so it runs FIRST (outermost middleware).
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        request.state.start_ts = time.time()

        if not isinstance(getattr(request.state, "route", None), str):
            request.state.route = request.url.path
        if not isinstance(getattr(request.state, "model_id", None), str):
            request.state.model_id = "unknown"
        if not isinstance(getattr(request.state, "cached", None), bool):
            request.state.cached = False

        return await call_next(request)


def _effective_model_load_mode(settings: Any) -> str:
    raw = getattr(settings, "model_load_mode", None)
    if isinstance(raw, str) and raw.strip():
        m = raw.strip().lower()
        return "eager" if m == "on" else m

    env = str(getattr(settings, "env", "dev")).strip().lower()
    return "eager" if env == "prod" else "lazy"


def _model_warmup_enabled(settings: Any, mode: str) -> bool:
    raw = os.getenv("MODEL_WARMUP")
    if raw is not None:
        return raw.strip().lower() in ("1", "true", "yes", "y", "on")

    env = str(getattr(settings, "env", "dev")).strip().lower()
    is_prod = env == "prod"
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

    s = getattr(app.state, "settings", None) or get_settings()
    app.state.settings = s

    # --------------------
    # Policy snapshot startup
    # --------------------
    try:
        snap = load_policy_decision_from_env()
        app.state.policy_snapshot = snap

        logging.getLogger("uvicorn.error").info(
            "policy: ok=%s source=%s model_id=%s enable_extract=%s error=%s",
            getattr(snap, "ok", None),
            getattr(snap, "source_path", None),
            getattr(snap, "model_id", None),
            getattr(snap, "enable_extract", None),
            getattr(snap, "error", None),
        )
    except Exception as e:
        app.state.policy_snapshot = None
        logging.getLogger("uvicorn.error").exception("Policy snapshot init failed (continuing): %s", e)

    mode = _effective_model_load_mode(s)

    logging.getLogger("uvicorn.error").info(
        "startup: env=%s debug=%s app_profile=%s db_instance=%s redis_enabled=%s model_load_mode=%s",
        getattr(s, "env", "dev"),
        getattr(s, "debug", False),
        os.getenv("APP_PROFILE", ""),
        getattr(s, "db_instance", "unknown"),
        getattr(s, "redis_enabled", False),
        mode,
    )

    logging.getLogger("uvicorn.error").info(
        "CORS allow_origins=%s",
        getattr(s, "cors_allowed_origins", ["*"]),
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
                if not hasattr(llm, "ensure_loaded"):
                    app.state.model_loaded = False
                    app.state.model_error = "LLM backend has no ensure_loaded(); cannot eager load"
                else:
                    llm.ensure_loaded()
                    app.state.model_loaded = True

                    if _model_warmup_enabled(s, mode):
                        try:
                            prompt = _warmup_prompt()
                            max_new = _warmup_max_new_tokens()

                            warm_backend = llm
                            if hasattr(llm, "default") and callable(getattr(llm, "default")):
                                warm_backend = llm.default()

                            _ = warm_backend.generate(prompt=prompt, max_new_tokens=max_new, temperature=0.0)
                        except Exception as e:
                            app.state.model_loaded = False
                            app.state.model_error = f"warmup_failed: {repr(e)}"
                            logging.getLogger("uvicorn.error").exception("Model warmup failed: %s", e)
                            logging.getLogger("uvicorn.error").error("Aborting startup: warmup failed in eager mode")
                            raise

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

    app.state.settings = s

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

    app.add_middleware(RequestContextMiddleware)

    from llm_server.api import health, generate, models, admin, extract

    app.include_router(health.router)
    app.include_router(generate.router)
    app.include_router(models.router)
    app.include_router(admin.router)
    app.include_router(extract.router)

    return app


app = create_app()