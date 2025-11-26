# src/llm_server/api/health.py
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Request
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from llm_server.core.config import settings
from llm_server.db.session import get_session
from llm_server.api.generate import get_llm

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
    Readiness probe used by tests.

    - Checks DB connectivity (SELECT 1).
    - Checks that the LLM is "loadable" via get_llm(request).ensure_loaded().
    """
    # DB check
    db_ok = True
    try:
        await session.execute(text("SELECT 1"))
    except Exception:  # pragma: no cover
        db_ok = False
        logger.exception("DB readiness check failed")

    # LLM check
    try:
        llm = get_llm(request)

        # Both ModelManager and MultiModelManager expose ensure_loaded()
        if hasattr(llm, "ensure_loaded"):
            llm.ensure_loaded()

        llm_status = "ready"
    except Exception:  # pragma: no cover
        llm_status = "not ready"
        logger.exception("LLM readiness check failed")

    overall = "ready" if db_ok and llm_status == "ready" else "not ready"

    return {
        "status": overall,               # tests expect "ready" or "ok"
        "db": "ok" if db_ok else "error",
        "llm": llm_status,
    }