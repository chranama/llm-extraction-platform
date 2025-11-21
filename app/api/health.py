from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Request, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_session
from app.services.llm import ModelManager

router = APIRouter()
_llm = ModelManager()


@router.get("/healthz")
async def healthz() -> dict:
    """
    Liveness probe: process is up and routing works.
    Do not perform heavy checks here.
    """
    return {"status": "ok"}


@router.get("/readyz")
async def readyz(
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """
    Readiness probe: check core dependencies we care about:
      - LLM loaded (ensure tokenizer/model available)
      - DB reachable (simple SELECT 1)
      - Redis reachable (PING), if configured
    """
    status = {
        "status": "ready",
        "llm": "ok",
        "db": "ok",
        "redis": "disabled",
    }

    # LLM
    try:
        _llm.ensure_loaded()
    except Exception as e:
        status["llm"] = f"error: {type(e).__name__}"
        status["status"] = "not ready"

    # DB
    try:
        await session.execute(select(1))
    except Exception as e:
        status["db"] = f"error: {type(e).__name__}"
        status["status"] = "not ready"

    # Redis (optional)
    redis = getattr(request.app.state, "redis", None)
    if redis is not None:
        try:
            pong = await redis.ping()
            status["redis"] = "ok" if pong else "error: ping falsey"
            if not pong:
                status["status"] = "not ready"
        except Exception as e:
            status["redis"] = f"error: {type(e).__name__}"
            status["status"] = "not ready"

    return status