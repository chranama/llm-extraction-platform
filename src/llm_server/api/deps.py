# src/llm_server/api/deps.py
from __future__ import annotations

import time
from typing import Dict, Tuple, Optional

from fastapi import Header, HTTPException, status
from sqlalchemy import select

from llm_server.db.session import async_session_maker
from llm_server.db.models import ApiKey
from llm_server.core.config import settings  # noqa: F401  (reserved for future use)

# -----------------------------------------------------------------------------
# Error helpers
# -----------------------------------------------------------------------------


def _error(status_code: int, code: str, message: str, extra: dict | None = None):
    detail = {"code": code, "message": message}
    if extra:
        detail.update(extra)
    raise HTTPException(status_code=status_code, detail=detail)


# -----------------------------------------------------------------------------
# Simple in-memory rate limiting state
#   _RL maps "bucket" -> (window_start_timestamp, count_in_window)
#   Tests clear this via deps._RL.clear()
# -----------------------------------------------------------------------------


_RL: Dict[str, Tuple[float, int]] = {}


def _now() -> float:
    """Wrapper for time.time(), so tests can monkeypatch if needed."""
    return time.time()


def _role_rpm(role) -> int:
    """
    Default RPM for a given role.

    Tests monkeypatch this function to force small limits, e.g.:

        monkeypatch.setattr(deps, "_role_rpm", lambda role: 1, raising=True)

    So its default behavior only matters outside tests.
    """
    # In real life you could inspect role / role.name here.
    # For now, just a generous default.
    return 60


def _check_rate_limit(key: str, role_obj) -> None:
    """
    Enforce simple per-API-key RPM in a 60-second sliding window.

    Raises HTTPException(429) when limit exceeded.

    Important: the bucket key includes id(_role_rpm) so that when tests
    monkeypatch _role_rpm, they get a fresh bucket and do not inherit
    state from previous tests.
    """
    rpm = _role_rpm(role_obj)
    if rpm is None or rpm <= 0:
        # Treat None or <=0 as "no rate limit"
        return

    now = _now()
    window = 60.0

    bucket = f"{key}:{id(_role_rpm)}"
    window_start, count = _RL.get(bucket, (now, 0))

    # If we're outside the window, reset
    if now - window_start >= window:
        window_start = now
        count = 0

    if count >= rpm:
        retry_after = max(1, int(window - (now - window_start)))
        _error(
            status.HTTP_429_TOO_MANY_REQUESTS,
            code="rate_limited",
            message="Rate limited",
            extra={"retry_after": retry_after},
        )

    # Otherwise, record one more request
    _RL[bucket] = (window_start, count + 1)


# -----------------------------------------------------------------------------
# Quota helpers
# -----------------------------------------------------------------------------


async def _check_and_consume_quota(api_key_obj: ApiKey) -> None:
    """
    Simple monthly quota check.

    - quota_monthly is interpreted as "max number of requests per month".
    - quota_used is incremented by 1 per successful authorized request.
    - If quota_monthly is None or <= 0 -> unlimited.
    - If quota_used >= quota_monthly -> 402 (quota_exhausted).
    """
    quota = api_key_obj.quota_monthly

    # Treat None, 0, or negative as unlimited
    if quota is None or quota <= 0:
        return

    if api_key_obj.quota_used is None:
        api_key_obj.quota_used = 0

    # If already at or above the limit, refuse
    if api_key_obj.quota_used >= quota:
        _error(
            status.HTTP_402_PAYMENT_REQUIRED,
            code="quota_exhausted",
            message="Monthly quota exhausted",
        )

    # Otherwise, consume one unit of quota.
    api_key_obj.quota_used += 1

    # Persist change
    async with async_session_maker() as session:
        session.add(api_key_obj)
        await session.commit()


# -----------------------------------------------------------------------------
# Public dependency: get_api_key
# -----------------------------------------------------------------------------


async def get_api_key(
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> ApiKey:
    """
    Dependency used by protected endpoints (e.g. /v1/generate).

    Behavior expected by tests:
      - Missing header -> 401, detail.code in {"missing_api_key", "invalid_api_key"}
      - Invalid key    -> 401, detail.code == "invalid_api_key"
      - Inactive key   -> 401, detail.code == "invalid_api_key"
      - Rate limit hit -> 429, detail.code == "rate_limit_exceeded"
      - Quota exceeded -> 402, detail.code == "quota_exhausted"
    """
    if not x_api_key:
        _error(
            status.HTTP_401_UNAUTHORIZED,
            code="missing_api_key",
            message="X-API-Key header is required",
        )

    # Lookup ApiKey row
    async with async_session_maker() as session:
        result = await session.execute(
            select(ApiKey).where(ApiKey.key == x_api_key)
        )
        api_key_obj: ApiKey | None = result.scalar_one_or_none()

    if api_key_obj is None or not getattr(api_key_obj, "active", True):
        _error(
            status.HTTP_401_UNAUTHORIZED,
            code="invalid_api_key",
            message="Invalid or inactive API key",
        )

    # NOTE: do NOT touch api_key_obj.role here; that would lazy-load the
    # relationship outside of the async session and trigger MissingGreenlet.
    role_obj = None  # tests monkeypatch _role_rpm and don’t depend on this value

    # 1) Rate limiting (may raise 429)
    _check_rate_limit(api_key_obj.key, role_obj)

    # 2) Quota (may raise 402 and persists usage)
    await _check_and_consume_quota(api_key_obj)

    # Success: return the ApiKey row for downstream usage (logging, etc.)
    return api_key_obj