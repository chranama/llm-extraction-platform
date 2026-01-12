# src/llm_server/api/admin.py
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Any, Dict

from fastapi import APIRouter, Depends, Query, Request, status
from pydantic import BaseModel, ConfigDict
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload, selectinload

from llm_server.api.deps import get_api_key
from llm_server.core.errors import AppError
from llm_server.db.models import ApiKey, InferenceLog, RoleTable
from llm_server.db.session import get_session

import asyncio

from llm_server.core.config import settings
from llm_server.services.llm import build_llm_from_settings, MultiModelManager

logger = logging.getLogger("llm_server.api.admin")

router = APIRouter(tags=["admin"])

_MODEL_LOAD_LOCK = asyncio.Lock()

# -------------------------------------------------------------------
# Models for responses
# -------------------------------------------------------------------


class MeUsageResponse(BaseModel):
    api_key: str
    role: Optional[str]
    total_requests: int
    first_request_at: Optional[datetime]
    last_request_at: Optional[datetime]
    total_prompt_tokens: int
    total_completion_tokens: int


class AdminUsageRow(BaseModel):
    api_key: str
    name: Optional[str]
    role: Optional[str]
    total_requests: int
    total_prompt_tokens: int
    total_completion_tokens: int
    first_request_at: Optional[datetime]
    last_request_at: Optional[datetime]


class AdminUsageResponse(BaseModel):
    results: List[AdminUsageRow]


class AdminApiKeyInfo(BaseModel):
    key_prefix: str
    name: Optional[str]
    role: Optional[str]
    created_at: datetime
    disabled: bool


class AdminApiKeyListResponse(BaseModel):
    results: List[AdminApiKeyInfo]
    total: int
    limit: int
    offset: int


class AdminLogEntry(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime

    api_key: Optional[str] = None
    route: str
    client_host: Optional[str] = None

    model_id: str
    latency_ms: Optional[float] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None

    # Full payloads (admin-only)
    prompt: str
    output: Optional[str] = None


class AdminLogsPage(BaseModel):
    total: int
    limit: int
    offset: int
    items: List[AdminLogEntry]


class AdminModelStats(BaseModel):
    model_id: str
    total_requests: int
    total_prompt_tokens: int
    total_completion_tokens: int
    avg_latency_ms: float | None


class AdminStatsResponse(BaseModel):
    window_days: int
    since: datetime
    total_requests: int
    total_prompt_tokens: int
    total_completion_tokens: int
    avg_latency_ms: float | None
    per_model: list[AdminModelStats]

class AdminLoadModelRequest(BaseModel):
    # optional: override default model id for this process
    model_id: Optional[str] = None


class AdminLoadModelResponse(BaseModel):
    ok: bool
    already_loaded: bool
    default_model: str
    models: list[str]


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


def _tag_admin_request(request: Request, route: str) -> None:
    # For metrics/logging middleware consistency
    request.state.route = route
    request.state.model_id = "admin"
    request.state.cached = False


async def _ensure_admin(api_key: ApiKey, session: AsyncSession) -> None:
    """
    Reload the ApiKey with its Role in the current async session and
    enforce that the caller has the 'admin' role.
    """
    result = await session.execute(
        select(ApiKey).options(joinedload(ApiKey.role)).where(ApiKey.id == api_key.id)
    )
    db_key = result.scalar_one_or_none()

    role_name = db_key.role.name if db_key and db_key.role else None
    if role_name != "admin":
        raise AppError(
            code="forbidden",
            message="Admin privileges required",
            status_code=status.HTTP_403_FORBIDDEN,
        )


# -------------------------------------------------------------------
# /v1/me/usage
# -------------------------------------------------------------------


@router.get("/v1/me/usage", response_model=MeUsageResponse)
async def get_my_usage(
    request: Request,
    api_key: ApiKey = Depends(get_api_key),
    session: AsyncSession = Depends(get_session),
):
    _tag_admin_request(request, "/v1/me/usage")

    # Fetch role name without lazy-loading problems
    role_row = await session.get(RoleTable, api_key.role_id) if api_key.role_id else None
    role_name = role_row.name if role_row else None

    # Aggregate this key's usage from inference_logs
    stmt = (
        select(
            func.count(InferenceLog.id),
            func.min(InferenceLog.created_at),
            func.max(InferenceLog.created_at),
            func.coalesce(func.sum(InferenceLog.prompt_tokens), 0),
            func.coalesce(func.sum(InferenceLog.completion_tokens), 0),
        )
        .where(InferenceLog.api_key == api_key.key)
    )
    res = await session.execute(stmt)
    (
        total_requests,
        first_request_at,
        last_request_at,
        total_prompt_tokens,
        total_completion_tokens,
    ) = res.one()

    return MeUsageResponse(
        api_key=api_key.key,
        role=role_name,
        total_requests=int(total_requests or 0),
        first_request_at=first_request_at,
        last_request_at=last_request_at,
        total_prompt_tokens=int(total_prompt_tokens or 0),
        total_completion_tokens=int(total_completion_tokens or 0),
    )


# -------------------------------------------------------------------
# /v1/admin/usage
# -------------------------------------------------------------------


@router.get("/v1/admin/usage", response_model=AdminUsageResponse)
async def get_admin_usage(
    request: Request,
    api_key: ApiKey = Depends(get_api_key),
    session: AsyncSession = Depends(get_session),
):
    _tag_admin_request(request, "/v1/admin/usage")
    await _ensure_admin(api_key, session)

    # Aggregate stats per api_key
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
    key_values = [r[0] for r in rows if r[0] is not None]
    key_map: dict[str, ApiKey] = {}
    if key_values:
        keys_stmt = (
            select(ApiKey)
            .options(selectinload(ApiKey.role))
            .where(ApiKey.key.in_(key_values))
        )
        key_objs = (await session.execute(keys_stmt)).scalars().all()
        key_map = {k.key: k for k in key_objs}

    results: List[AdminUsageRow] = []
    for key_value, total_requests, first_at, last_at, total_prompt, total_completion in rows:
        key_obj = key_map.get(key_value)
        results.append(
            AdminUsageRow(
                api_key=key_value,
                name=getattr(key_obj, "name", None) if key_obj else None,
                role=key_obj.role.name if key_obj and key_obj.role else None,
                total_requests=int(total_requests or 0),
                total_prompt_tokens=int(total_prompt or 0),
                total_completion_tokens=int(total_completion or 0),
                first_request_at=first_at,
                last_request_at=last_at,
            )
        )

    return AdminUsageResponse(results=results)


# -------------------------------------------------------------------
# /v1/admin/keys
# -------------------------------------------------------------------


@router.get("/v1/admin/keys", response_model=AdminApiKeyListResponse)
async def list_api_keys(
    request: Request,
    api_key: ApiKey = Depends(get_api_key),
    session: AsyncSession = Depends(get_session),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """
    List API keys and their metadata.

    - Admin-only
    - Does NOT return full key values, only a key prefix for identification.
    """
    _tag_admin_request(request, "/v1/admin/keys")
    await _ensure_admin(api_key, session)

    # Total count
    total_stmt = select(func.count(ApiKey.id))
    total = (await session.execute(total_stmt)).scalar_one()
    total_int = int(total or 0)

    # Page of keys, eager-load role to avoid lazy-load issues
    stmt = (
        select(ApiKey)
        .options(selectinload(ApiKey.role))
        .order_by(ApiKey.created_at.desc())
        .offset(offset)
        .limit(limit)
    )

    keys = (await session.execute(stmt)).scalars().all()

    results: List[AdminApiKeyInfo] = []
    for k in keys:
        prefix = k.key[:8] if k.key else ""
        disabled_flag = bool(getattr(k, "disabled_at", None))

        results.append(
            AdminApiKeyInfo(
                key_prefix=prefix,
                name=getattr(k, "name", None),
                role=k.role.name if k.role else None,
                created_at=k.created_at,
                disabled=disabled_flag,
            )
        )

    return AdminApiKeyListResponse(
        results=results,
        total=total_int,
        limit=limit,
        offset=offset,
    )


# -------------------------------------------------------------------
# /v1/admin/logs
# -------------------------------------------------------------------


@router.get("/v1/admin/logs", response_model=AdminLogsPage)
async def list_inference_logs(
    request: Request,
    api_key: ApiKey = Depends(get_api_key),
    session: AsyncSession = Depends(get_session),
    # Filters
    model_id: Optional[str] = Query(default=None, description="Filter by model_id"),
    key: Optional[str] = Query(default=None, alias="api_key", description="Filter by API key value"),
    route: Optional[str] = Query(default=None, description="Filter by route, e.g. /v1/generate"),
    from_ts: Optional[datetime] = Query(default=None, description="Filter logs created_at >= this timestamp (ISO8601)"),
    to_ts: Optional[datetime] = Query(default=None, description="Filter logs created_at <= this timestamp (ISO8601)"),
    limit: int = Query(default=50, ge=1, le=200, description="Max number of rows to return"),
    offset: int = Query(default=0, ge=0, description="Offset for pagination"),
):
    """
    Admin-only: list inference logs with basic filters + pagination.
    """
    _tag_admin_request(request, "/v1/admin/logs")
    await _ensure_admin(api_key, session)

    filters = []

    if model_id:
        filters.append(InferenceLog.model_id == model_id)
    if key:
        filters.append(InferenceLog.api_key == key)
    if route:
        filters.append(InferenceLog.route == route)
    if from_ts:
        filters.append(InferenceLog.created_at >= from_ts)
    if to_ts:
        filters.append(InferenceLog.created_at <= to_ts)

    # ---- total count ----
    count_stmt = select(func.count()).select_from(InferenceLog)
    if filters:
        count_stmt = count_stmt.where(*filters)

    total = await session.scalar(count_stmt)
    total = int(total or 0)

    # ---- page query ----
    stmt = select(InferenceLog)
    if filters:
        stmt = stmt.where(*filters)

    stmt = stmt.order_by(InferenceLog.created_at.desc()).offset(offset).limit(limit)

    result = await session.execute(stmt)
    rows = result.scalars().all()

    # Pydantic v2
    items = [AdminLogEntry.model_validate(row) for row in rows]

    return AdminLogsPage(
        total=total,
        limit=limit,
        offset=offset,
        items=items,
    )


# -------------------------------------------------------------------
# /v1/admin/stats
# -------------------------------------------------------------------


@router.get("/v1/admin/stats", response_model=AdminStatsResponse)
async def get_admin_stats(
    request: Request,
    api_key: ApiKey = Depends(get_api_key),
    session: AsyncSession = Depends(get_session),
    window_days: int = Query(30, ge=1, le=365),
):
    """
    Global usage stats over a sliding time window (admin-only).

    - Aggregate totals from `inference_logs`
    - Per-model breakdown
    """
    _tag_admin_request(request, "/v1/admin/stats")
    await _ensure_admin(api_key, session)

    now = datetime.now(timezone.utc)
    since = now - timedelta(days=window_days)

    # ---- Global aggregates ----
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

    # ---- Per-model aggregates ----
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
    per_model_items: list[AdminModelStats] = []

    for mid, count, p_tokens, c_tokens, m_avg_latency in per_model_rows:
        per_model_items.append(
            AdminModelStats(
                model_id=mid,
                total_requests=int(count or 0),
                total_prompt_tokens=int(p_tokens or 0),
                total_completion_tokens=int(c_tokens or 0),
                avg_latency_ms=float(m_avg_latency) if m_avg_latency is not None else None,
            )
        )

    return AdminStatsResponse(
        window_days=window_days,
        since=since,
        total_requests=int(total_requests or 0),
        total_prompt_tokens=int(total_prompt_tokens or 0),
        total_completion_tokens=int(total_completion_tokens or 0),
        avg_latency_ms=float(avg_latency_ms) if avg_latency_ms is not None else None,
        per_model=per_model_items,
    )

# -------------------------------------------------------------------
# /v1/admin/models/load
# -------------------------------------------------------------------

@router.post("/v1/admin/models/load", response_model=AdminLoadModelResponse)
async def admin_load_model(
    request: Request,
    body: AdminLoadModelRequest,
    api_key: ApiKey = Depends(get_api_key),
    session: AsyncSession = Depends(get_session),
):
    """
    Admin-only: explicitly load the LLM into app.state.llm.

    Intended for dev: start stack with MODEL_LOAD_MODE=off or lazy, then call this endpoint.
    """
    _tag_admin_request(request, "/v1/admin/models/load")
    await _ensure_admin(api_key, session)

    async with _MODEL_LOAD_LOCK:
        app = request.app

        # If already loaded and healthy, just report
        existing = getattr(app.state, "llm", None)
        model_loaded = bool(getattr(app.state, "model_loaded", False))
        model_error = getattr(app.state, "model_error", None)

        if existing is not None and model_loaded and not model_error:
            if isinstance(existing, MultiModelManager):
                model_ids = list(existing.models.keys())
                default_model = existing.default_id
            else:
                default_model = getattr(existing, "model_id", settings.model_id)
                model_ids = [default_model]

            return AdminLoadModelResponse(
                ok=True,
                already_loaded=True,
                default_model=default_model,
                models=model_ids,
            )

        # Validate optional override (must be allowed by settings/models.yaml)
        if body.model_id:
            if body.model_id not in settings.all_model_ids:
                raise AppError(
                    code="model_not_allowed",
                    message=f"Model '{body.model_id}' not allowed.",
                    status_code=status.HTTP_400_BAD_REQUEST,
                    extra={"allowed": settings.all_model_ids},
                )
            settings.model_id = body.model_id  # type: ignore[attr-defined]

        # Reset authoritative flags before attempting
        app.state.model_error = None
        app.state.model_loaded = False

        # NOTE: we intentionally DO NOT change model_load_mode here.
        # mode=="off" means "no autoload on startup", not "cannot be loaded".
        try:
            llm = build_llm_from_settings()
            app.state.llm = llm

            # "load" endpoint should guarantee weights are in memory
            if hasattr(llm, "ensure_loaded"):
                llm.ensure_loaded()
                app.state.model_loaded = True
            else:
                # If backend cannot be loaded, report not loaded
                app.state.model_loaded = False

        except Exception as e:
            app.state.model_error = repr(e)
            app.state.model_loaded = False
            app.state.llm = None
            raise

        # Report models
        llm = app.state.llm
        if isinstance(llm, MultiModelManager):
            model_ids = list(llm.models.keys())
            default_model = llm.default_id
        else:
            default_model = getattr(llm, "model_id", settings.model_id)
            model_ids = [default_model]

        return AdminLoadModelResponse(
            ok=True,
            already_loaded=False,
            default_model=default_model,
            models=model_ids,
        )