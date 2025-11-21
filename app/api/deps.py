from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, Any

from fastapi import Depends, Header, HTTPException, Request
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.session import get_session
from app.db.models import ApiKey, RoleTable, Role

# -----------------------
# Helpers & local caches
# -----------------------

# Fallback in-memory TTL cache for key metadata (used if Redis is disabled/unavailable)
# key -> (payload_dict, expires_epoch_seconds)
_KEYCACHE: Dict[str, Tuple[dict, float]] = {}
_KEYCACHE_TTL = 10.0  # seconds

def _now_s() -> float:
    return time.time()

def _utcnow_naive() -> datetime:
    # Use naive UTC to avoid tz-aware vs tz-naive comparison errors
    return datetime.utcnow()

def _role_rpm(role_name: Optional[str]) -> int:
    if role_name == Role.admin.value:
        return settings.rate_limit_rpm_admin
    if role_name == Role.free.value:
        return settings.rate_limit_rpm_free
    return settings.rate_limit_rpm_default

# -----------------------
# DB helpers
# -----------------------

async def _load_key(session: AsyncSession, raw_key: str) -> Optional[ApiKey]:
    q = select(ApiKey).where(ApiKey.key == raw_key, ApiKey.active == True)  # noqa: E712
    res = await session.execute(q)
    return res.scalar_one_or_none()

async def _maybe_reset_quota(session: AsyncSession, row: ApiKey) -> ApiKey:
    """If monthly quota window expired, reset counters on the row."""
    if row.quota_monthly is None:
        return row
    now = _utcnow_naive()
    reset_at = row.quota_reset_at  # may be None or naive
    need_reset = reset_at is None or now >= reset_at
    if need_reset:
        next_reset = now + timedelta(days=settings.quota_auto_reset_days)
        await session.execute(
            update(ApiKey)
            .where(ApiKey.id == row.id)
            .values(quota_used=0, quota_reset_at=next_reset)
        )
        await session.commit()
        row.quota_used = 0
        row.quota_reset_at = next_reset
    return row

async def _enforce_and_increment_quota(session: AsyncSession, row: ApiKey):
    """Increment monthly usage if not exhausted; else raise 402-like error."""
    if row.quota_monthly is None:
        return

    stmt = (
        update(ApiKey)
        .where(ApiKey.id == row.id)
        .where(ApiKey.quota_used < ApiKey.quota_monthly)
        .values(quota_used=ApiKey.quota_used + 1)
    )
    res = await session.execute(stmt)
    if res.rowcount != 1:
        reset_in = None
        if row.quota_reset_at is not None:
            reset_in = max(0, int((row.quota_reset_at - _utcnow_naive()).total_seconds()))
        raise HTTPException(
            status_code=402,  # Payment Required (semantically close)
            detail={
                "code": "quota_exhausted",
                "message": "Monthly quota exhausted.",
                "reset_in_seconds": reset_in,
            },
        )
    await session.commit()

# -----------------------
# Redis helpers (optional)
# -----------------------

def _get_redis(request: Request):
    # app.state.redis is set in main.py startup; may be None
    return getattr(request.app.state, "redis", None)

async def _redis_get_json(redis, key: str) -> Optional[dict]:
    if not redis:
        return None
    raw = await redis.get(key)
    return json.loads(raw) if raw else None

async def _redis_set_json(redis, key: str, value: dict, ttl_s: int):
    if not redis:
        return
    await redis.set(key, json.dumps(value, default=str), ex=ttl_s)

async def _redis_rpm_check(redis, api_key: str, rpm: int) -> Optional[int]:
    """
    Fixed-window per-minute rate limit:
    - returns None if allowed,
    - returns retry_after_seconds (int) if blocked.
    """
    if not redis or rpm <= 0:
        return None  # rpm<=0 means unlimited

    # Window key like: rl:{key}:{epoch_minute}
    now = int(time.time())
    window = now // 60
    rkey = f"rl:{api_key}:{window}"

    # atomically increment and set expiry 60s
    current = await redis.incr(rkey)
    if current == 1:
        await redis.expire(rkey, 60)

    if current > rpm:
        # Compute TTL remaining
        ttl = await redis.ttl(rkey)
        if ttl is None or ttl < 0:
            ttl = 60
        return int(ttl)
    return None

# -----------------------
# Public dependency
# -----------------------

async def require_api_key(
    request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    session: AsyncSession = Depends(get_session),
) -> ApiKey:
    """
    Validate API key, enforce role-based RPM (Redis if available), and monthly quota (DB).
    Returns the ApiKey ORM row for downstream use.
    """
    if not x_api_key:
        raise HTTPException(status_code=401, detail={"code": "missing_api_key", "message": "X-API-Key required"})

    redis = _get_redis(request)
    now = _now_s()

    # 1) Load key metadata (Redis cache -> DB -> in-memory fallback cache)
    row: Optional[ApiKey] = None
    cached_obj: Optional[dict] = None

    # Try Redis cached snapshot
    if redis:
        cached_obj = await _redis_get_json(redis, f"apikey:{x_api_key}:obj")

    if cached_obj is None:
        # Try in-memory cache
        entry = _KEYCACHE.get(x_api_key)
        if entry and entry[1] > now:
            cached_obj = entry[0]

    if cached_obj is not None and cached_obj.get("id"):
        # Rehydrate a lightweight ApiKey-like object
        row = ApiKey(
            id=cached_obj["id"],
            key=cached_obj["key"],
            label=cached_obj.get("label"),
            active=cached_obj.get("active", True),
            quota_monthly=cached_obj.get("quota_monthly"),
            quota_used=cached_obj.get("quota_used", 0),
            quota_reset_at=cached_obj.get("quota_reset_at"),
            role_id=cached_obj.get("role_id"),
        )
        # NOTE: .role relationship is not populated in this snapshot path (we only need role name for RPM).
        role_name = cached_obj.get("role_name")
    else:
        # DB hit
        row = await _load_key(session, x_api_key)
        if not row:
            # Negative-cache briefly
            _KEYCACHE[x_api_key] = ({}, now + 3.0)
            raise HTTPException(status_code=401, detail={"code": "invalid_api_key", "message": "Invalid/inactive key"})

        # Eager load role name (avoid lazy load later)
        role_name = None
        if row.role_id:
            # best-effort: fetch role name; ignore if missing
            role_row = await session.get(RoleTable, row.role_id)
            role_name = role_row.name if role_row else None

        # Maybe reset quota
        row = await _maybe_reset_quota(session, row)

        # Cache a portable snapshot
        snapshot = {
            "id": row.id,
            "key": row.key,
            "label": row.label,
            "active": row.active,
            "quota_monthly": row.quota_monthly,
            "quota_used": row.quota_used,
            "quota_reset_at": row.quota_reset_at.isoformat() if row.quota_reset_at else None,
            "role_id": row.role_id,
            "role_name": role_name,
        }
        # Redis cache (short TTL)
        if redis:
            await _redis_set_json(redis, f"apikey:{x_api_key}:obj", snapshot, ttl_s=int(_KEYCACHE_TTL))
        # In-memory cache
        _KEYCACHE[x_api_key] = (snapshot, now + _KEYCACHE_TTL)

    # 2) Rate limit (per-minute by role) — Redis first, fallback to in-memory fixed window
    rpm = _role_rpm(role_name)
    if rpm > 0:
        # Try Redis rate limit
        retry_after = await _redis_rpm_check(redis, x_api_key, rpm)
        if retry_after is None and not redis:
            # Fallback in-memory fixed window limiter
            # Very small/simple: keep per-process counts; OK for dev
            window_end = int((now // 60 + 1) * 60)
            # in-memory store as: _RL = { key: (count, reset_epoch) }
            global _RL
            try:
                _RL
            except NameError:
                _RL = {}
            count, reset_at = _RL.get(x_api_key, (0, window_end))
            if now > reset_at:
                count, reset_at = 0, window_end
            if count + 1 > rpm:
                raise HTTPException(
                    status_code=429,
                    detail={"code": "rate_limited", "message": "RPM limit exceeded.", "retry_after_seconds": int(reset_at - now)},
                )
            _RL[x_api_key] = (count + 1, reset_at)
        elif retry_after is not None:
            raise HTTPException(
                status_code=429,
                detail={"code": "rate_limited", "message": "RPM limit exceeded.", "retry_after_seconds": retry_after},
            )

    # 3) Monthly quota (DB) — enforce and increment
    await _enforce_and_increment_quota(session, row)

    # 4) Attach request context for logging/metrics
    request.state.api_key = x_api_key
    request.state.api_key_label = row.label or "unlabeled"
    request.state.api_key_role = role_name

    return row