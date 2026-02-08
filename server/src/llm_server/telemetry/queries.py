# server/src/llm_server/telemetry/queries.py
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload, selectinload

from llm_server.db.models import ApiKey, InferenceLog, RoleTable
from llm_server.telemetry.types import (
    AdminStats,
    AdminUsageRow,
    ApiKeyInfo,
    ApiKeyListPage,
    LogsPage,
    MeUsage,
    ModelStats,
)

# -----------------------------
# Small helpers
# -----------------------------


def _pct(values: list[float], p: float) -> float | None:
    """
    Simple percentile on a list of floats.

    - p in [0, 100]
    - returns None if no values
    """
    if not values:
        return None
    if p <= 0:
        return float(min(values))
    if p >= 100:
        return float(max(values))

    xs = sorted(values)
    # "nearest rank" style
    k = int(round((p / 100.0) * (len(xs) - 1)))
    k = max(0, min(len(xs) - 1, k))
    return float(xs[k])


def _safe_int(x: Any) -> int:
    try:
        return int(x or 0)
    except Exception:
        return 0


def _safe_float(x: Any) -> float | None:
    try:
        return float(x) if x is not None else None
    except Exception:
        return None


# -----------------------------
# Existing API (back-compat)
# -----------------------------


async def fetch_role_name(session: AsyncSession, role_id: int | None) -> Optional[str]:
    if not role_id:
        return None
    role_row = await session.get(RoleTable, role_id)
    return role_row.name if role_row else None


async def get_me_usage(session: AsyncSession, *, api_key_value: str, role_name: Optional[str]) -> MeUsage:
    """
    Back-compat: counts ALL requests for that api key (success + failure).
    Tokens will naturally sum over successes (failures have NULL tokens).
    """
    stmt = (
        select(
            func.count(InferenceLog.id),
            func.min(InferenceLog.created_at),
            func.max(InferenceLog.created_at),
            func.coalesce(func.sum(InferenceLog.prompt_tokens), 0),
            func.coalesce(func.sum(InferenceLog.completion_tokens), 0),
        )
        .where(InferenceLog.api_key == api_key_value)
    )
    res = await session.execute(stmt)
    (total_requests, first_request_at, last_request_at, total_prompt_tokens, total_completion_tokens) = res.one()

    return MeUsage(
        api_key=api_key_value,
        role=role_name,
        total_requests=_safe_int(total_requests),
        first_request_at=first_request_at,
        last_request_at=last_request_at,
        total_prompt_tokens=_safe_int(total_prompt_tokens),
        total_completion_tokens=_safe_int(total_completion_tokens),
    )


async def get_admin_usage(session: AsyncSession) -> list[AdminUsageRow]:
    """
    Back-compat: aggregates ALL requests per api_key.

    NOTE:
      - Failures are now logged with api_key possibly "" (empty) depending on your deps.
      - This function preserves original behavior and does not attempt to separate errors,
        because AdminUsageRow doesn't have error fields.
    """
    stmt = (
        select(
            InferenceLog.api_key,
            func.count(InferenceLog.id),
            func.min(InferenceLog.created_at),
            func.max(InferenceLog.created_at),
            func.coalesce(func.sum(InferenceLog.prompt_tokens), 0),
            func.coalesce(func.sum(InferenceLog.completion_tokens), 0),
        )
        .group_by(InferenceLog.api_key)
    )

    rows = (await session.execute(stmt)).all()

    # Fetch key metadata in one shot
    key_values = [r[0] for r in rows if isinstance(r[0], str) and r[0]]
    key_map: dict[str, ApiKey] = {}
    if key_values:
        keys_stmt = select(ApiKey).options(selectinload(ApiKey.role)).where(ApiKey.key.in_(key_values))
        key_objs = (await session.execute(keys_stmt)).scalars().all()
        key_map = {k.key: k for k in key_objs}

    out: list[AdminUsageRow] = []
    for key_value, total_requests, first_at, last_at, total_prompt, total_completion in rows:
        key_obj = key_map.get(key_value) if isinstance(key_value, str) else None
        out.append(
            AdminUsageRow(
                api_key=key_value,
                name=getattr(key_obj, "name", None) if key_obj else None,
                role=key_obj.role.name if key_obj and key_obj.role else None,
                total_requests=_safe_int(total_requests),
                total_prompt_tokens=_safe_int(total_prompt),
                total_completion_tokens=_safe_int(total_completion),
                first_request_at=first_at,
                last_request_at=last_at,
            )
        )

    return out


async def list_api_keys(session: AsyncSession, *, limit: int, offset: int) -> ApiKeyListPage:
    total_stmt = select(func.count(ApiKey.id))
    total = (await session.execute(total_stmt)).scalar_one()
    total_int = _safe_int(total)

    stmt = (
        select(ApiKey)
        .options(selectinload(ApiKey.role))
        .order_by(ApiKey.created_at.desc())
        .offset(offset)
        .limit(limit)
    )

    keys = (await session.execute(stmt)).scalars().all()

    items: list[ApiKeyInfo] = []
    for k in keys:
        prefix = k.key[:8] if k.key else ""
        disabled_flag = bool(getattr(k, "disabled_at", None))
        items.append(
            ApiKeyInfo(
                key_prefix=prefix,
                name=getattr(k, "name", None),
                role=k.role.name if k.role else None,
                created_at=k.created_at,
                disabled=disabled_flag,
            )
        )

    return ApiKeyListPage(total=total_int, limit=limit, offset=offset, items=items)


async def list_inference_logs(
    session: AsyncSession,
    *,
    model_id: Optional[str],
    api_key_value: Optional[str],
    route: Optional[str],
    from_ts: Optional[datetime],
    to_ts: Optional[datetime],
    limit: int,
    offset: int,
    # NEW optional filters (won't break existing callers)
    status_code_min: int | None = None,
    status_code_max: int | None = None,
    error_code: str | None = None,
    error_stage: str | None = None,
    cached: bool | None = None,
) -> LogsPage:
    filters = []

    if model_id:
        filters.append(InferenceLog.model_id == model_id)
    if api_key_value:
        filters.append(InferenceLog.api_key == api_key_value)
    if route:
        filters.append(InferenceLog.route == route)
    if from_ts:
        filters.append(InferenceLog.created_at >= from_ts)
    if to_ts:
        filters.append(InferenceLog.created_at <= to_ts)

    # new expanded fields
    if status_code_min is not None:
        filters.append(InferenceLog.status_code >= int(status_code_min))
    if status_code_max is not None:
        filters.append(InferenceLog.status_code <= int(status_code_max))
    if error_code:
        filters.append(InferenceLog.error_code == error_code)
    if error_stage:
        filters.append(InferenceLog.error_stage == error_stage)
    if isinstance(cached, bool):
        filters.append(InferenceLog.cached == cached)

    # total
    count_stmt = select(func.count()).select_from(InferenceLog)
    if filters:
        count_stmt = count_stmt.where(*filters)

    total = await session.scalar(count_stmt)
    total_int = _safe_int(total)

    # page
    stmt = select(InferenceLog)
    if filters:
        stmt = stmt.where(*filters)

    stmt = stmt.order_by(InferenceLog.created_at.desc()).offset(offset).limit(limit)
    rows = (await session.execute(stmt)).scalars().all()

    return LogsPage(total=total_int, limit=limit, offset=offset, items=list(rows))


async def get_admin_stats(session: AsyncSession, *, window_days: int) -> AdminStats:
    """
    Back-compat AdminStats:
      - counts ALL requests in window (success + failure)
      - token sums naturally reflect successes only
      - avg_latency includes failures where latency_ms is present (it should be, now)
    """
    now = datetime.now(timezone.utc)
    since = now - timedelta(days=window_days)

    global_stmt = (
        select(
            func.count(InferenceLog.id),
            func.coalesce(func.sum(InferenceLog.prompt_tokens), 0),
            func.coalesce(func.sum(InferenceLog.completion_tokens), 0),
            func.avg(InferenceLog.latency_ms),
        )
        .where(InferenceLog.created_at >= since)
    )

    total_requests, total_prompt_tokens, total_completion_tokens, avg_latency_ms = (
        await session.execute(global_stmt)
    ).one()

    per_model_stmt = (
        select(
            InferenceLog.model_id,
            func.count(InferenceLog.id),
            func.coalesce(func.sum(InferenceLog.prompt_tokens), 0),
            func.coalesce(func.sum(InferenceLog.completion_tokens), 0),
            func.avg(InferenceLog.latency_ms),
        )
        .where(InferenceLog.created_at >= since)
        .group_by(InferenceLog.model_id)
    )

    per_model_rows = (await session.execute(per_model_stmt)).all()
    per_model_items: list[ModelStats] = []
    for mid, count, p_tokens, c_tokens, m_avg_latency in per_model_rows:
        per_model_items.append(
            ModelStats(
                model_id=mid,
                total_requests=_safe_int(count),
                total_prompt_tokens=_safe_int(p_tokens),
                total_completion_tokens=_safe_int(c_tokens),
                avg_latency_ms=_safe_float(m_avg_latency),
            )
        )

    return AdminStats(
        window_days=window_days,
        since=since,
        total_requests=_safe_int(total_requests),
        total_prompt_tokens=_safe_int(total_prompt_tokens),
        total_completion_tokens=_safe_int(total_completion_tokens),
        avg_latency_ms=_safe_float(avg_latency_ms),
        per_model=per_model_items,
    )


async def reload_key_with_role(session: AsyncSession, *, api_key_id: int) -> ApiKey | None:
    """
    Utility used by API layer for admin gating without lazy-load issues.
    """
    result = await session.execute(select(ApiKey).options(joinedload(ApiKey.role)).where(ApiKey.id == api_key_id))
    return result.scalar_one_or_none()


# -----------------------------
# NEW: SLO-ish window summaries
# -----------------------------


async def compute_window_slo_snapshot(
    session: AsyncSession,
    *,
    window_seconds: int,
    routes: list[str] | None = None,
    model_id: str | None = None,
) -> dict[str, Any]:
    """
    Compute a small SLO snapshot from InferenceLog (DB-source-of-truth).

    Returned dict is intentionally schema-friendly (you can map it to a contracts artifact later):
      - totals: requests, errors, error_rate
      - latency_ms: avg, p95
      - tokens: prompt_total, completion_total, completion_p95 (success-only)
      - dimensions: window_seconds, since, routes, model_id

    NOTE: Percentiles are computed in Python for portability across SQLite/Postgres/DuckDB.
    """
    now = datetime.now(timezone.utc)
    since = now - timedelta(seconds=int(window_seconds))

    filters = [InferenceLog.created_at >= since]
    if routes:
        filters.append(InferenceLog.route.in_(routes))
    if model_id:
        filters.append(InferenceLog.model_id == model_id)

    # totals + error count in SQL (fast)
    totals_stmt = select(
        func.count(InferenceLog.id).label("n"),
        func.coalesce(func.sum(case((InferenceLog.status_code >= 400, 1), else_=0)), 0).label("errors"),
        func.coalesce(func.sum(InferenceLog.prompt_tokens), 0).label("prompt_tokens"),
        func.coalesce(func.sum(InferenceLog.completion_tokens), 0).label("completion_tokens"),
        func.avg(InferenceLog.latency_ms).label("avg_latency_ms"),
    ).where(*filters)

    row = (await session.execute(totals_stmt)).one()
    n = _safe_int(row.n)
    errors = _safe_int(row.errors)

    # values for percentiles (portable)
    lat_stmt = select(InferenceLog.latency_ms).where(*filters).where(InferenceLog.latency_ms.is_not(None))
    latencies = [float(x[0]) for x in (await session.execute(lat_stmt)).all() if x and x[0] is not None]

    # completion tokens percentile (success-only: completion_tokens non-null)
    ct_stmt = (
        select(InferenceLog.completion_tokens)
        .where(*filters)
        .where(InferenceLog.completion_tokens.is_not(None))
    )
    completion_tokens_vals = [float(x[0]) for x in (await session.execute(ct_stmt)).all() if x and x[0] is not None]

    error_rate = (errors / n) if n > 0 else 0.0

    return {
        "window_seconds": int(window_seconds),
        "window_end": now,
        "since": since,
        "routes": list(routes or []),
        "model_id": model_id,
        "totals": {
            "requests": n,
            "errors": errors,
            "error_rate": error_rate,
        },
        "latency_ms": {
            "avg": _safe_float(row.avg_latency_ms),
            "p95": _pct(latencies, 95.0),
        },
        "tokens": {
            "prompt_total": _safe_int(row.prompt_tokens),
            "completion_total": _safe_int(row.completion_tokens),
            "completion_p95": _pct(completion_tokens_vals, 95.0),
        },
    }