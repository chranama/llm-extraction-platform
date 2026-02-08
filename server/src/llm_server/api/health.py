# server/src/llm_server/api/health.py
from __future__ import annotations

import logging
from typing import Any, Tuple

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from llm_server.core.config import get_settings
from llm_server.db.session import get_session
from llm_server.core.redis import get_redis_from_request

logger = logging.getLogger("llm_server.api.health")
router = APIRouter(tags=["health"])


def _settings(request: Request):
    return getattr(request.app.state, "settings", None) or get_settings()


def _effective_model_load_mode(request: Request) -> str:
    mode = getattr(request.app.state, "model_load_mode", None)
    if isinstance(mode, str) and mode:
        return mode
    return str(_settings(request).model_load_mode)


def _llm_state(llm) -> str:
    if llm is None:
        return "not initialized"

    fn = getattr(llm, "is_loaded", None)
    if callable(fn):
        try:
            return "loaded" if fn() else "not loaded"
        except Exception:
            return "unknown"

    for attr in ("loaded", "is_ready", "ready"):
        if hasattr(llm, attr):
            try:
                return "loaded" if bool(getattr(llm, attr)) else "not loaded"
            except Exception:
                return "unknown"

    for attr in ("model", "_model", "pipeline", "_pipeline"):
        if hasattr(llm, attr):
            try:
                return "loaded" if getattr(llm, attr) is not None else "not loaded"
            except Exception:
                return "unknown"

    return "unknown"


async def _db_check(session: AsyncSession) -> Tuple[bool, str]:
    try:
        await session.execute(text("SELECT 1"))
        return True, "ok"
    except Exception:
        logger.exception("DB check failed")
        return False, "error"


async def _redis_check(request: Request) -> Tuple[bool, str]:
    s = _settings(request)
    if not s.redis_enabled:
        return True, "disabled"

    try:
        redis = get_redis_from_request(request)
        if redis is None:
            return False, "not initialized"

        pong = await redis.ping()
        return (pong is True), ("ok" if pong is True else f"unexpected: {pong}")
    except Exception:
        logger.exception("Redis check failed")
        return False, "error"


def _model_required_for_readyz(request: Request) -> bool:
    s = _settings(request)
    return bool(s.require_model_ready)


@router.get("/healthz")
async def healthz():
    """
    Liveness probe.
    Must always be fast and must never fail due to infra or model state.
    """
    return {"status": "ok"}


@router.get("/readyz")
async def readyz(
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """
    Readiness probe.
    Determines whether the service can receive traffic.
    """
    s = _settings(request)

    db_ok, db_status = await _db_check(session)
    redis_ok, redis_status = await _redis_check(request)

    llm = getattr(request.app.state, "llm", None)
    model_loaded = bool(getattr(request.app.state, "model_loaded", False))
    model_error = getattr(request.app.state, "model_error", None)

    mode = _effective_model_load_mode(request)
    llm_status = "disabled" if mode == "off" else _llm_state(llm)

    require_model = _model_required_for_readyz(request)

    model_ok = True
    if require_model:
        model_ok = model_loaded and model_error is None

    ready = db_ok and redis_ok and model_ok

    payload = {
        "status": "ready" if ready else "not ready",
        "db": db_status,
        "redis": redis_status,
        "db_instance": getattr(s, "db_instance", "unknown"),
        "model_load_mode": mode,
        "require_model_ready": require_model,
        "model_loaded": model_loaded,
        "model_error": model_error,
        "llm": llm_status,
    }
    return JSONResponse(payload, status_code=200 if ready else 503)


@router.get("/modelz")
async def modelz(request: Request):
    """
    Model-only readiness.
    Never triggers loading.
    """
    s = _settings(request)

    llm = getattr(request.app.state, "llm", None)
    mode = _effective_model_load_mode(request)

    model_loaded = bool(getattr(request.app.state, "model_loaded", False))
    model_error = getattr(request.app.state, "model_error", None)
    llm_status = "disabled" if mode == "off" else _llm_state(llm)

    ready = model_loaded and model_error is None

    payload = {
        "status": "ready" if ready else "not ready",
        "db_instance": getattr(s, "db_instance", "unknown"),
        "model_load_mode": mode,
        "model_loaded": model_loaded,
        "model_error": model_error,
        "llm": llm_status,
    }
    return JSONResponse(payload, status_code=200 if ready else 503)