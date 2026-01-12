# src/llm_server/api/health.py
from __future__ import annotations

import os
import logging
from typing import Optional, Tuple

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
    default = "eager" if settings.env.strip().lower() == "prod" else "lazy"
    return os.getenv("MODEL_LOAD_MODE", default).strip().lower()


def get_llm_from_request(request: Request):
    """
    Canonical place to grab the LLM instance.
    Keeping it here avoids importing from api.generate.
    """
    return getattr(request.app.state, "llm", None)


def _llm_state(llm) -> str:
    """
    Best-effort inference of whether the model is actually loaded.

    We avoid calling ensure_loaded() here because /readyz and /modelz
    should not *cause* a load unless you explicitly choose to elsewhere.
    """
    if llm is None:
        return "not initialized"

    # Common patterns:
    # - wrappers expose is_loaded()
    # - wrappers expose loaded boolean
    # - wrappers store underlying model at .model when loaded
    for attr in ("is_loaded",):
        fn = getattr(llm, attr, None)
        if callable(fn):
            try:
                return "loaded" if bool(fn()) else "not loaded"
            except Exception:
                logger.exception("LLM is_loaded() check failed")
                return "unknown"

    for attr in ("loaded", "is_ready", "ready"):
        if hasattr(llm, attr):
            try:
                return "loaded" if bool(getattr(llm, attr)) else "not loaded"
            except Exception:
                logger.exception("LLM %s attribute check failed", attr)
                return "unknown"

    # Heuristic: an underlying model handle exists only when loaded
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
    Service readiness: "is the API ready to serve *non-LLM* traffic reliably?"
    Checks DB and (optionally) Redis.

    IMPORTANT: This endpoint does NOT trigger model loading.
    It reports model status but does not require it to be loaded.
    """
    db_ok, db_status = await _db_readiness(session)
    redis_ok, redis_status = await _redis_readiness(request)

    mode = _model_load_mode()
    llm = get_llm_from_request(request)
    llm_status = "disabled" if mode == "off" else _llm_state(llm)

    overall_ready = db_ok and redis_ok
    payload = {
        "status": "ready" if overall_ready else "not ready",
        "db": db_status,
        "redis": redis_status,
        "model_load_mode": mode,
        "llm": llm_status,  # informational only for /readyz
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
    model_loaded = bool(getattr(app.state, "model_loaded", False))

    # If we have a recorded error, it's not ready regardless of mode
    if model_error:
        payload = {
            "status": "not ready",
            "model_load_mode": mode,
            "model_loaded": model_loaded,
            "model_error": model_error,
        }
        return JSONResponse(content=payload, status_code=503)

    # MODE=off means "no autoload", but modelz should still reflect whether
    # the admin endpoint has loaded it yet.
    if mode == "off":
        payload = {
            "status": "ready" if model_loaded else "not ready",
            "model_load_mode": mode,
            "model_loaded": model_loaded,
            "model_error": None,
        }
        return JSONResponse(content=payload, status_code=200 if model_loaded else 503)

    # lazy/eager/on: same contract—ready iff loaded flag is true
    payload = {
        "status": "ready" if model_loaded else "not ready",
        "model_load_mode": mode,
        "model_loaded": model_loaded,
        "model_error": None,
    }
    return JSONResponse(content=payload, status_code=200 if model_loaded else 503)