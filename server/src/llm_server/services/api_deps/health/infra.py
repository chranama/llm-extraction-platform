# server/src/llm_server/services/api_deps/health/infra.py
from __future__ import annotations

import logging
from typing import Tuple

from fastapi import Request
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from llm_server.core.redis import get_redis_from_request
from llm_server.core.config import get_settings

logger = logging.getLogger("llm_server.api_deps.health.infra")


def settings_from_request(request: Request):
    return getattr(request.app.state, "settings", None) or get_settings()


async def db_check(session: AsyncSession) -> Tuple[bool, str]:
    try:
        await session.execute(text("SELECT 1"))
        return True, "ok"
    except Exception:
        logger.exception("DB check failed")
        return False, "error"


async def redis_check(request: Request) -> Tuple[bool, str]:
    s = settings_from_request(request)
    if not bool(getattr(s, "redis_enabled", False)):
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