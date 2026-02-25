# server/src/llm_server/main.py
from __future__ import annotations

import os
import secrets
import time
from contextlib import asynccontextmanager
from typing import Any

import anyio
import orjson
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from llm_server.core import errors, limits, logging as logging_config, metrics
from llm_server.core.config import get_settings
from llm_server.core.redis import close_redis, init_redis
from llm_server.io.policy_decisions import load_policy_decision_from_env
from llm_server.services.llm_runtime.llm_build import build_llm_from_settings
from llm_server.services.llm_runtime.llm_config import load_models_config
from llm_server.services.llm_runtime.llm_loader import RuntimeModelLoader
from llm_server.services.limits.early_reject_middleware import EarlyRejectGenerateMiddleware

# NEW: config contracts validation
from llm_contracts.config import validate_assessment_for_extract, validate_models_config

_REQUEST_ID_HEADER = "X-Request-ID"


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        request.state.start_ts = time.time()

        rid = request.headers.get(_REQUEST_ID_HEADER) or request.headers.get("x-request-id")
        if not (isinstance(rid, str) and rid.strip()):
            rid = secrets.token_hex(16)
        request.state.request_id = rid

        if not isinstance(getattr(request.state, "route", None), str):
            request.state.route = request.url.path
        if not isinstance(getattr(request.state, "model_id", None), str):
            request.state.model_id = "unknown"
        if not isinstance(getattr(request.state, "cached", None), bool):
            request.state.cached = False

        resp = await call_next(request)

        try:
            if _REQUEST_ID_HEADER not in resp.headers:
                resp.headers[_REQUEST_ID_HEADER] = rid
        except Exception:
            pass

        return resp


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


def _cors_origins(settings: Any) -> list[str]:
    raw = getattr(settings, "cors_allowed_origins", None)
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []

    out: list[str] = []
    for x in raw:
        if not isinstance(x, str):
            continue
        s = x.strip()
        if s:
            out.append(s)
    return out


async def _warmup_generate_offloop(model_backend: Any, *, prompt: str, max_new_tokens: int) -> None:
    def _run() -> None:
        gen = getattr(model_backend, "generate", None)
        if callable(gen):
            _ = gen(prompt=prompt, max_new_tokens=max_new_tokens, temperature=0.0)

    try:
        await anyio.to_thread.run_sync(_run)
    except Exception:
        return


def _validate_models_config_or_raise(cfg: Any, *, mode: str) -> None:
    """
    Validate normalized models_config using llm_contracts.config.

    - Always validates structural shape.
    - Validates Option A assessed semantics (require_for_extract => per-model assessed boolean).
    """
    # Structural validation
    r1 = validate_models_config(cfg, allow_generic_deployment_key=(os.getenv("ALLOW_GENERIC_DEPLOYMENT_KEY", "").strip() == "1"))
    if not r1.ok:
        raise RuntimeError(f"models_config invalid: {r1.error}")

    # Assessed semantics validation (Option A)
    r2 = validate_assessment_for_extract(cfg)
    if not r2.ok:
        raise RuntimeError(f"models_config assessment invalid: {r2.error}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    import logging

    s = getattr(app.state, "settings", None) or get_settings()
    app.state.settings = s

    # ---- policy snapshot (best-effort) ----
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

    origins = _cors_origins(s)
    logging.getLogger("uvicorn.error").info("CORS enabled=%s allow_origins=%s", bool(origins), origins)

    # ---- redis ----
    try:
        app.state.redis = await init_redis()
    except Exception as e:
        logging.getLogger("uvicorn.error").exception("Redis init failed: %s", e)
        app.state.redis = None

    # ---- runtime state ----
    app.state.model_load_mode = mode
    app.state.model_error = None
    app.state.model_loaded = False
    app.state.runtime_default_model_id = None

    # ---- models config + llm registry (NO WEIGHTS) ----
    try:
        cfg = load_models_config()
        # NEW: validate immediately after load, before any registry build depends on it
        _validate_models_config_or_raise(cfg, mode=mode)
        app.state.models_config = cfg
    except Exception as e:
        app.state.models_config = None
        app.state.model_error = repr(e)
        logging.getLogger("uvicorn.error").exception("models_config init failed: %s", e)

        # Match your “abort in eager” behavior (models config is foundational)
        if mode in ("eager", "on"):
            raise

    try:
        app.state.llm = build_llm_from_settings()
    except Exception as e:
        app.state.llm = None
        app.state.model_error = repr(e)

        if mode in ("eager", "on"):
            logging.getLogger("uvicorn.error").exception("LLM registry init failed; aborting startup: %s", e)
            raise

        logging.getLogger("uvicorn.error").exception("LLM registry init failed (lazy/off continues): %s", e)

    # ---- runtime loader (the ONLY explicit weight-loading path) ----
    app.state.runtime_model_loader = RuntimeModelLoader(app.state)

    # ---- eager load path (explicit) ----
    if mode in ("eager", "on"):
        try:
            r = await app.state.runtime_model_loader.load_default()
            app.state.model_loaded = bool(r.loaded)

            if _model_warmup_enabled(s, mode):
                prompt = _warmup_prompt()
                max_new = _warmup_max_new_tokens()

                llm = getattr(app.state, "llm", None)
                warm_backend = llm
                try:
                    if hasattr(llm, "default") and callable(getattr(llm, "default")):
                        warm_backend = llm.default()
                except Exception:
                    warm_backend = llm

                if warm_backend is not None:
                    await _warmup_generate_offloop(warm_backend, prompt=prompt, max_new_tokens=max_new)

        except Exception as e:
            app.state.model_error = repr(e)
            app.state.model_loaded = False
            logging.getLogger("uvicorn.error").exception("LLM eager init failed; aborting startup: %s", e)
            raise

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

    origins = _cors_origins(s)
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=[_REQUEST_ID_HEADER],
            max_age=600,
        )

    logging_config.setup(app)
    limits.setup(app)
    metrics.setup(app)
    errors.setup(app)

    app.add_middleware(EarlyRejectGenerateMiddleware)
    app.add_middleware(RequestContextMiddleware)

    from llm_server.api import admin, extract, generate, health, models

    app.include_router(health.router)
    app.include_router(generate.router)
    app.include_router(models.router)
    app.include_router(admin.router)
    app.include_router(extract.router)

    return app


app = create_app()