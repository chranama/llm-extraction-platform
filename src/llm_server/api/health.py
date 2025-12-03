# src/llm_server/api/health.py
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Request
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from llm_server.core.config import settings
from llm_server.db.session import get_session
from llm_server.api.generate import get_llm
from llm_server.core.redis import get_redis_from_request

logger = logging.getLogger("llm_server.api.health")

router = APIRouter(tags=["health"])


@router.get("/healthz")
async def healthz():
    """
    Simple liveness check.
    """
    return {"status": "ok"}


@router.get("/readyz")
async def readyz(
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """
    Readiness probe.

    - Checks DB connectivity (SELECT 1).
    - Checks that the LLM is "loadable" via get_llm(request).ensure_loaded().
    - Optionally checks Redis when settings.redis_enabled is True.
    """
    # ---------- DB check ----------
    db_ok = True
    try:
        await session.execute(text("SELECT 1"))
    except Exception:  # pragma: no cover
        db_ok = False
        logger.exception("DB readiness check failed")

    # ---------- LLM check ----------
    try:
        llm = get_llm(request)

        # Both ModelManager and MultiModelManager expose ensure_loaded()
        if hasattr(llm, "ensure_loaded"):
            llm.ensure_loaded()

        llm_status = "ready"
    except Exception:  # pragma: no cover
        llm_status = "not ready"
        logger.exception("LLM readiness check failed")

    # ---------- Redis check (optional) ----------
    if settings.redis_enabled:
        redis_ok = True
        redis_status_str = "ok"

        try:
            redis = get_redis_from_request(request)
            if redis is None:
                redis_ok = False
                redis_status_str = "not initialized"
                logger.error(
                    "Redis enabled in settings, but app.state.redis is None"
                )
            else:
                pong = await redis.ping()
                if pong is not True:
                    redis_ok = False
                    redis_status_str = f"unexpected response: {pong}"
                    logger.error("Redis ping returned unexpected value: %s", pong)
        except Exception:  # pragma: no cover
            redis_ok = False
            redis_status_str = "error"
            logger.exception("Redis readiness check failed")
    else:
        # Redis not required for readiness
        redis_ok = True
        redis_status_str = "disabled"

    # ---------- Overall status ----------
    overall_ready = db_ok and llm_status == "ready" and redis_ok
    overall_status = "ready" if overall_ready else "not ready"

    return {
        "status": overall_status,              # tests expect "ready" or "ok"-style
        "db": "ok" if db_ok else "error",
        "llm": llm_status,                     # "ready" or "not ready"
        "redis": redis_status_str,             # "ok" | "disabled" | "not initialized" | "error"
    }