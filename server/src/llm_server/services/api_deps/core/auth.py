# server/src/llm_server/services/api_deps/core/auth.py
from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

from fastapi import Depends, Header, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from llm_server.core.errors import AppError
from llm_server.db.models import ApiKey
from llm_server.db.session import get_session

_RL: Dict[str, Tuple[float, int]] = {}


def clear_rate_limit_state() -> None:
    _RL.clear()


def _now() -> float:
    return time.time()


def _role_rpm(role_obj: Any) -> int:
    # Placeholder role-based logic.
    # Keep it deterministic and fast; can be replaced later by DB/role config.
    return 60


def _check_rate_limit(key: str, role_obj: Any) -> None:
    rpm = _role_rpm(role_obj)
    if rpm is None or rpm <= 0:
        return

    now = _now()
    window = 60.0
    bucket = f"{key}:{id(_role_rpm)}"
    window_start, count = _RL.get(bucket, (now, 0))

    if now - window_start >= window:
        window_start = now
        count = 0

    if count >= rpm:
        retry_after = max(1, int(window - (now - window_start)))
        raise AppError(
            code="rate_limited",
            message="Rate limited",
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            extra={"retry_after": retry_after},
        )

    _RL[bucket] = (window_start, count + 1)


def _check_and_consume_quota_in_session(api_key_obj: ApiKey) -> None:
    quota = api_key_obj.quota_monthly
    if quota is None or quota <= 0:
        return

    if api_key_obj.quota_used is None:
        api_key_obj.quota_used = 0

    if api_key_obj.quota_used >= quota:
        raise AppError(
            code="quota_exhausted",
            message="Monthly quota exhausted",
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
        )

    api_key_obj.quota_used += 1


async def get_api_key(
    request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    session: AsyncSession = Depends(get_session),
) -> ApiKey:
    """
    Authn + simple local RPM limiter + quota consumption.

    Notes:
      - Rate limiting here is in-memory (per-process). If you need distributed rate limiting,
        move this to Redis later.
      - Quota consumption happens with a commit (consistent with your current behavior).
    """
    if not x_api_key:
        raise AppError(
            code="missing_api_key",
            message="X-API-Key header is required",
            status_code=status.HTTP_401_UNAUTHORIZED,
        )

    result = await session.execute(select(ApiKey).where(ApiKey.key == x_api_key))
    api_key_obj: ApiKey | None = result.scalar_one_or_none()

    if api_key_obj is None or not getattr(api_key_obj, "active", True):
        raise AppError(
            code="invalid_api_key",
            message="Invalid or inactive API key",
            status_code=status.HTTP_401_UNAUTHORIZED,
        )

    try:
        request.state.api_key = api_key_obj.key
    except Exception:
        pass

    role_obj = None
    _check_rate_limit(api_key_obj.key, role_obj)

    _check_and_consume_quota_in_session(api_key_obj)
    session.add(api_key_obj)
    await session.commit()

    return api_key_obj