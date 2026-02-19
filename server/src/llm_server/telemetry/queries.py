# server/src/llm_server/telemetry/queries.py
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload, selectinload

from llm_contracts.runtime.generate_slo import RUNTIME_GENERATE_SLO_VERSION
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
    Simple percentile on a list of floats (portable).

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


def _utc_iso_z() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


# -----------------------------
# Existing API (unchanged)
# -----------------------------


async def fetch_role_name(session: AsyncSession, role_id: int | None) -> Optional[str]:
    if not role_id:
        return None
    role_row = await session.get(RoleTable, role_id)
    return role_row.name if role_row else None


async def get_me_usage(session: AsyncSession, *, api_key_value: str, role_name: Optional[str]) -> MeUsage:
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
    (
        total_requests,
        first_request_at,
        last_request_at,
        total_prompt_tokens,
        total_completion_tokens,
    ) = res.one()

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

    key_values = [r[0] for r in rows if isinstance(r[0], str) and r[0]]
    key_map: dict[str, ApiKey] = {}
    if key_values:
        keys_stmt = (
            select(ApiKey)
            .options(selectinload(ApiKey.role))
            .where(ApiKey.key.in_(key_values))
        )
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

    return ApiKeyListPage(
        total=_safe_int(total),
        limit=limit,
        offset=offset,
        items=items,
    )


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

    count_stmt = select(func.count()).select_from(InferenceLog)
    if filters:
        count_stmt = count_stmt.where(*filters)

    total = await session.scalar(count_stmt)

    stmt = select(InferenceLog)
    if filters:
        stmt = stmt.where(*filters)

    stmt = stmt.order_by(InferenceLog.created_at.desc()).offset(offset).limit(limit)
    rows = (await session.execute(stmt)).scalars().all()

    return LogsPage(
        total=_safe_int(total),
        limit=limit,
        offset=offset,
        items=list(rows),
    )


async def get_admin_stats(session: AsyncSession, *, window_days: int) -> AdminStats:
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
    result = await session.execute(
        select(ApiKey)
        .options(joinedload(ApiKey.role))
        .where(ApiKey.id == api_key_id)
    )
    return result.scalar_one_or_none()


# -----------------------------
# NEW (A7): contracts-shaped SLO
# -----------------------------


async def compute_window_generate_slo_contracts_payload(
    session: AsyncSession,
    *,
    window_seconds: int,
    routes: list[str] | None = None,
    model_id: str | None = None,
) -> dict[str, Any]:
    """
    Compute a runtime_generate_slo_v1 payload directly from DB telemetry.

    This payload:
      - validates against runtime_generate_slo_v1.schema.json
      - is readable by llm_contracts.runtime.generate_slo
      - is suitable for policy generate-clamp decisions
    """
    now = datetime.now(timezone.utc)
    since = now - timedelta(seconds=int(window_seconds))

    filters = [InferenceLog.created_at >= since]
    if routes:
        filters.append(InferenceLog.route.in_(routes))
    if model_id:
        filters.append(InferenceLog.model_id == model_id)

    totals_stmt = select(
        func.count(InferenceLog.id).label("n"),
        func.coalesce(func.sum(case((InferenceLog.status_code >= 400, 1), else_=0)), 0).label("errors"),
        func.avg(InferenceLog.latency_ms).label("avg_latency_ms"),
        func.max(InferenceLog.latency_ms).label("max_latency_ms"),
        func.avg(InferenceLog.prompt_tokens).label("prompt_avg"),
        func.max(InferenceLog.prompt_tokens).label("prompt_max"),
        func.avg(InferenceLog.completion_tokens).label("completion_avg"),
        func.max(InferenceLog.completion_tokens).label("completion_max"),
    ).where(*filters)

    row = (await session.execute(totals_stmt)).one()
    n = _safe_int(row.n)
    errors = _safe_int(row.errors)
    error_rate = (errors / n) if n > 0 else 0.0

    lat_stmt = (
        select(InferenceLog.latency_ms)
        .where(*filters)
        .where(InferenceLog.latency_ms.is_not(None))
    )
    latencies = [float(x[0]) for x in (await session.execute(lat_stmt)).all() if x and x[0] is not None]

    ct_stmt = (
        select(InferenceLog.completion_tokens)
        .where(*filters)
        .where(InferenceLog.completion_tokens.is_not(None))
    )
    completion_vals = [float(x[0]) for x in (await session.execute(ct_stmt)).all() if x and x[0] is not None]

    ts = _utc_iso_z()
    routes_out = list(routes or ["/v1/generate", "/v1/generate/batch"])

    totals_obj = {
        "requests": {"total": int(n)},
        "errors": {
            "total": int(errors),
            "rate": float(error_rate),
            "by_status": {},
            "by_code": {},
        },
        "latency_ms": {
            "p50": float(_pct(latencies, 50.0) or 0.0),
            "p95": float(_pct(latencies, 95.0) or 0.0),
            "p99": float(_pct(latencies, 99.0) or 0.0),
            "avg": float(_safe_float(row.avg_latency_ms) or 0.0),
            "max": float(_safe_float(row.max_latency_ms) or 0.0),
        },
        "tokens": {
            "prompt": {
                "avg": float(_safe_float(row.prompt_avg) or 0.0),
                "max": int(_safe_int(row.prompt_max)),
            },
            "completion": {
                "avg": float(_safe_float(row.completion_avg) or 0.0),
                "p95": float(_pct(completion_vals, 95.0) or 0.0),
                "max": int(_safe_int(row.completion_max)),
            },
        },
    }

    model_row = {
        "model_id": model_id or "mixed",
        "requests": totals_obj["requests"],
        "errors": totals_obj["errors"],
        "latency_ms": totals_obj["latency_ms"],
        "tokens": totals_obj["tokens"],
    }

    return {
        "schema_version": RUNTIME_GENERATE_SLO_VERSION,
        "generated_at": ts,
        "window_seconds": int(window_seconds),
        "window_end": ts,
        "routes": routes_out,
        "models": [model_row],
        "totals": totals_obj,
    }