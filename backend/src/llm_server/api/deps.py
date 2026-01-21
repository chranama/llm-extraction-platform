from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

from fastapi import Depends, Header, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from llm_server.core.errors import AppError
from llm_server.core.config import get_settings
from llm_server.db.models import ApiKey
from llm_server.db.session import get_session
from llm_server.services.llm import build_llm_from_settings

# -----------------------------------------------------------------------------
# Simple in-memory rate limiting state
# -----------------------------------------------------------------------------
_RL: Dict[str, Tuple[float, int]] = {}


def _now() -> float:
    return time.time()


def _role_rpm(role_obj: Any) -> int:
    return 60


def _check_rate_limit(key: str, role_obj: Any) -> None:
    rpm = _role_rpm(role_obj)
    if rpm is None or rpm <= 0:
        return

    now = _now()
    window = 60.0

    # Bucket includes id(_role_rpm) so monkeypatching in tests doesn't share buckets.
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
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    session: AsyncSession = Depends(get_session),
) -> ApiKey:
    """
    Dependency used by protected endpoints.

      - Missing header -> AppError(401) code="missing_api_key"
      - Invalid key    -> AppError(401) code="invalid_api_key"
      - Rate limit hit -> AppError(429) code="rate_limited"
      - Quota exceeded -> AppError(402) code="quota_exhausted"
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

    # IMPORTANT: do NOT touch api_key_obj.role here if it's lazy; keep it None for now.
    role_obj = None

    _check_rate_limit(api_key_obj.key, role_obj)

    _check_and_consume_quota_in_session(api_key_obj)
    session.add(api_key_obj)
    await session.commit()

    return api_key_obj


def _effective_model_load_mode_from_request(request: Request) -> str:
    """
    Single source of truth for model mode:
      1) app.state.model_load_mode (set in lifespan)
      2) app.state.settings / get_settings()
    """
    mode = getattr(request.app.state, "model_load_mode", None)
    if isinstance(mode, str) and mode.strip():
        return mode.strip().lower()

    s = getattr(request.app.state, "settings", None) or get_settings()

    raw = getattr(s, "model_load_mode", None)
    if isinstance(raw, str) and raw.strip():
        return raw.strip().lower()

    env = str(getattr(s, "env", "dev")).strip().lower()
    return "eager" if env == "prod" else "lazy"


def get_llm(request: Request) -> Any:
    mode = _effective_model_load_mode_from_request(request)
    llm = getattr(request.app.state, "llm", None)

    # "off" means: do not build lazily on request
    if mode == "off":
        if llm is not None:
            return llm
        raise AppError(
            code="llm_not_loaded",
            message="LLM is not loaded. Call POST /v1/admin/models/load (MODEL_LOAD_MODE=off).",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    # lazy/eager/on: ensure llm object exists (loading semantics handled elsewhere)
    if llm is None:
        llm = build_llm_from_settings()
        request.app.state.llm = llm

    return llm