# src/llm_server/api/deps.py
from __future__ import annotations

import time
from typing import Dict, Tuple, Optional, Any

from fastapi import Header, status, Request
from sqlalchemy import select

from llm_server.core.errors import AppError
from llm_server.db.models import ApiKey
from llm_server.db.session import async_session_maker
from llm_server.services.llm import build_llm_from_settings
import os


# -----------------------------------------------------------------------------
# Simple in-memory rate limiting state
#   _RL maps "bucket" -> (window_start_timestamp, count_in_window)
#   Tests clear this via deps._RL.clear()
# -----------------------------------------------------------------------------

_RL: Dict[str, Tuple[float, int]] = {}


def _now() -> float:
    """Wrapper for time.time(), so tests can monkeypatch if needed."""
    return time.time()


def _role_rpm(role_obj: Any) -> int:
    """
    Default RPM for a given role.

    Tests may monkeypatch this function to force small limits, e.g.:
        monkeypatch.setattr(deps, "_role_rpm", lambda role: 1, raising=True)
    """
    return 60


def _check_rate_limit(key: str, role_obj: Any) -> None:
    """
    Enforce simple per-API-key RPM in a 60-second window.

    Raises AppError(429) when limit exceeded.

    Bucket includes id(_role_rpm) so monkeypatching in tests doesn't share buckets.
    """
    rpm = _role_rpm(role_obj)
    if rpm is None or rpm <= 0:
        return

    now = _now()
    window = 60.0

    bucket = f"{key}:{id(_role_rpm)}"
    window_start, count = _RL.get(bucket, (now, 0))

    # reset window if expired
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
    """
    Simple monthly quota check.

    - quota_monthly is "max number of requests per month"
    - quota_used increments by 1 per successful authorized request
    - quota_monthly is None or <=0 => unlimited
    """
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
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> ApiKey:
    """
    Dependency used by protected endpoints.

    New error scheme:
      - Missing header -> AppError(401) code="missing_api_key"
      - Invalid key    -> AppError(401) code="invalid_api_key"
      - Inactive key   -> AppError(401) code="invalid_api_key"
      - Rate limit hit -> AppError(429) code="rate_limited"
      - Quota exceeded -> AppError(402) code="quota_exhausted"
    """
    if not x_api_key:
        raise AppError(
            code="missing_api_key",
            message="X-API-Key header is required",
            status_code=status.HTTP_401_UNAUTHORIZED,
        )

    async with async_session_maker() as session:
        result = await session.execute(select(ApiKey).where(ApiKey.key == x_api_key))
        api_key_obj: ApiKey | None = result.scalar_one_or_none()

        if api_key_obj is None or not getattr(api_key_obj, "active", True):
            raise AppError(
                code="invalid_api_key",
                message="Invalid or inactive API key",
                status_code=status.HTTP_401_UNAUTHORIZED,
            )

        # IMPORTANT: do NOT touch api_key_obj.role here if it's a lazy relationship;
        # that can trigger MissingGreenlet if accessed outside an async session.
        role_obj = None

        # 1) Rate limiting (no DB writes)
        _check_rate_limit(api_key_obj.key, role_obj)

        # 2) Quota check + consume (DB write)
        _check_and_consume_quota_in_session(api_key_obj)
        session.add(api_key_obj)
        await session.commit()

        return api_key_obj
    
def get_llm(request: Request) -> Any:
    """
    FastAPI dependency for retrieving the in-process LLM backend.

    MODEL_LOAD_MODE semantics:
      - off  => never auto-load; only use if already loaded via admin endpoint
      - lazy => build on first request
      - eager => treated like lazy here (startup can pre-load if you implement it)
    """
    mode = os.getenv("MODEL_LOAD_MODE", "lazy").strip().lower()
    llm = getattr(request.app.state, "llm", None)

    if mode == "off":
        if llm is not None:
            return llm

        raise AppError(
            code="llm_not_loaded",
            message="LLM is not loaded. Call POST /v1/admin/models/load (MODEL_LOAD_MODE=off).",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    if llm is None:
        llm = build_llm_from_settings()
        request.app.state.llm = llm

    return llm