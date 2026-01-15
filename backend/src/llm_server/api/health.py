# src/llm_server/api/health.py
from __future__ import annotations

import os
import logging
from typing import Tuple

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from llm_server.core.config import settings
from llm_server.db.session import get_session
from llm_server.core.redis import get_redis_from_request

logger = logging.getLogger("llm_server.api.health")
router = APIRouter(tags=["health"])


def _model_load_mode() -> str:
    """
    Controls whether the application *tries* to load a model automatically elsewhere.
    This endpoint should never trigger loading.

    Values:
      - eager: load model at startup (recommended for prod)
      - lazy: load model on first request (ok for dev, not ideal for readiness)
      - off: do not auto-load (admin/manual only)
    """
    default = "eager" if settings.env.strip().lower() == "prod" else "lazy"
    return os.getenv("MODEL_LOAD_MODE", default).strip().lower()


def _require_model_loaded_for_readyz() -> bool:
    """
    Cloud-readiness policy:
      - In prod, require the default model to be loaded for /readyz by default.
      - Allow override via REQUIRE_MODEL_READY=0/1.
    """
    raw = os.getenv("REQUIRE_MODEL_READY")
    if raw is not None:
        return raw.strip().lower() in ("1", "true", "yes", "y", "on")
    # default policy
    return settings.env.strip().lower() == "prod"


def get_llm_from_request(request: Request):
    """
    Canonical place to grab the LLM instance.
    """
    return getattr(request.app.state, "llm", None)


def _llm_state(llm) -> str:
    """
    Best-effort inference of whether the model is actually loaded.

    Avoid calling ensure_loaded() here because readiness endpoints should not
    *cause* loading.
    """
    if llm is None:
        return "not initialized"

    # Prefer explicit is_loaded() if present
    fn = getattr(llm, "is_loaded", None)
    if callable(fn):
        try:
            return "loaded" if bool(fn()) else "not loaded"
        except Exception:
            logger.exception("LLM is_loaded() check failed")
            return "unknown"

    # Common boolean attrs
    for attr in ("loaded", "is_ready", "ready"):
        if hasattr(llm, attr):
            try:
                return "loaded" if bool(getattr(llm, attr)) else "not loaded"
            except Exception:
                logger.exception("LLM %s attribute check failed", attr)
                return "unknown"

    # Heuristic: underlying model handle exists only when loaded
    for attr in ("model", "_model", "pipeline", "_pipeline"):
        if hasattr(llm, attr):
            try:
                return "loaded" if getattr(llm, attr) is not None else "not loaded"
            except Exception:
                logger.exception("LLM %s heuristic check failed", attr)
                return "unknown"

    return "unknown"


async def _db_readiness(session: AsyncSession) -> Tuple[bool, str]:
    try:
        await session.execute(text("SELECT 1"))
        return True, "ok"
    except Exception:
        logger.exception("DB readiness check failed")
        return False, "error"


async def _redis_readiness(request: Request) -> Tuple[bool, str]:
    if not settings.redis_enabled:
        return True, "disabled"

    try:
        redis = get_redis_from_request(request)
        if redis is None:
            logger.error("Redis enabled but app.state.redis is None")
            return False, "not initialized"

        pong = await redis.ping()
        if pong is True:
            return True, "ok"

        logger.error("Redis ping unexpected: %s", pong)
        return False, f"unexpected response: {pong}"
    except Exception:
        logger.exception("Redis readiness check failed")
        return False, "error"


def _model_loaded_flag(request: Request) -> bool | None:
    """
    Prefer the explicit app.state.model_loaded flag if the app sets it during startup/warmup.
    Returns None if the flag isn't present.
    """
    app = request.app
    if hasattr(app.state, "model_loaded"):
        try:
            return bool(getattr(app.state, "model_loaded"))
        except Exception:
            return None
    return None


@router.get("/healthz")
async def healthz():
    """
    Liveness: "can I reach the process?"
    Must be fast and must not touch DB/Redis/model.
    """
    return {"status": "ok"}


@router.get("/readyz")
async def readyz(
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """
    Service readiness: "can I route real traffic here safely?"
    Checks DB and (optionally) Redis.

    Cloud policy (default):
      - In prod, also require the default model to be loaded.
      - In dev, model load is informational unless REQUIRE_MODEL_READY=1.

    IMPORTANT: This endpoint never triggers model loading.
    """
    db_ok, db_status = await _db_readiness(session)
    redis_ok, redis_status = await _redis_readiness(request)

    mode = _model_load_mode()
    llm = get_llm_from_request(request)

    # Prefer explicit state flag; fall back to llm inspection.
    loaded_flag = _model_loaded_flag(request)
    llm_status = "disabled" if mode == "off" else _llm_state(llm)
    model_loaded = (
        loaded_flag
        if loaded_flag is not None
        else (llm_status == "loaded")
    )

    require_model = _require_model_loaded_for_readyz()

    # If mode is off, we *can* still be ready for non-LLM traffic,
    # but if require_model is true, then "off" implies not ready until loaded.
    model_ok = True
    if require_model:
        model_ok = bool(model_loaded)

    overall_ready = db_ok and redis_ok and model_ok

    payload = {
        "status": "ready" if overall_ready else "not ready",
        "db": db_status,
        "redis": redis_status,
        "model_load_mode": mode,
        "require_model_ready": require_model,
        "model_loaded": bool(model_loaded),
        "llm": llm_status,  # informational detail
    }
    return JSONResponse(content=payload, status_code=200 if overall_ready else 503)


@router.get("/modelz")
async def modelz(request: Request):
    """
    Model readiness: "is the model loaded and immediately usable for inference?"
    Does NOT trigger loading.
    """
    app = request.app
    mode = getattr(app.state, "model_load_mode", None) or _model_load_mode()

    model_error = getattr(app.state, "model_error", None)

    loaded_flag = _model_loaded_flag(request)
    llm = get_llm_from_request(request)
    llm_status = "disabled" if mode == "off" else _llm_state(llm)

    model_loaded = (
        loaded_flag
        if loaded_flag is not None
        else (llm_status == "loaded")
    )

    if model_error:
        payload = {
            "status": "not ready",
            "model_load_mode": mode,
            "model_loaded": bool(model_loaded),
            "model_error": model_error,
            "llm": llm_status,
        }
        return JSONResponse(content=payload, status_code=503)

    payload = {
        "status": "ready" if model_loaded else "not ready",
        "model_load_mode": mode,
        "model_loaded": bool(model_loaded),
        "model_error": None,
        "llm": llm_status,
    }
    return JSONResponse(content=payload, status_code=200 if model_loaded else 503)