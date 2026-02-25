# server/src/llm_server/api/admin.py
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, cast

from fastapi import APIRouter, Depends, Query, Request, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession

from llm_server.core.config import get_settings
from llm_server.core.errors import AppError
from llm_server.db.models import ApiKey
from llm_server.db.session import get_session
from llm_server.io.policy_decisions import get_policy_snapshot, reload_policy_snapshot
from llm_server.io.runtime_generate_slo import write_generate_slo_artifact
from llm_server.reports import writer as report_w
from llm_server.services.api_deps.admin.authz import ensure_admin
from llm_server.services.api_deps.admin.models_ops import (
    allowed_model_ids_from_settings,
    get_loader,
    runtime_default_model_id_from_app,
    summarize_registry,
)
from llm_server.services.api_deps.admin.reload_ops import reload_runtime_state
from llm_server.services.api_deps.core.auth import get_api_key
from llm_server.services.api_deps.core.models_config import clear_models_config_cache
from llm_server.services.api_deps.core.policy_snapshot import snapshot_generate_cap
from llm_server.services.api_deps.core.settings import settings_from_request
from llm_server.services.api_deps.enforcement.capabilities import effective_capabilities
from llm_server.services.llm_runtime.inference import set_request_meta
from llm_server.telemetry import queries as telem_q

logger = logging.getLogger("llm_server.api.admin")
router = APIRouter(tags=["admin"])


# -------------------------------------------------------------------
# Models for responses (API contract)
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
    """
    Admin log row. Keep this stable: UI depends on it.
    New fields are optional for back-compat with older DBs/tests.
    """
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime

    api_key: Optional[str] = None
    request_id: Optional[str] = None
    route: str
    client_host: Optional[str] = None

    model_id: str

    # expanded telemetry fields
    status_code: Optional[int] = None
    cached: Optional[bool] = None
    error_code: Optional[str] = None
    error_stage: Optional[str] = None

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
    """
    Explicit weight-load request.

    For transformers, this will usually allocate weights.
    For external backends (e.g., llama-server), ensure_loaded() is typically a cheap probe/no-op.
    """
    model_id: Optional[str] = None


class AdminLoadModelResponse(BaseModel):
    ok: bool
    loaded: bool
    model_id: str
    load_mode: str
    detail: Dict[str, Any] = Field(default_factory=dict)


class AdminPolicySnapshotResponse(BaseModel):
    ok: bool
    model_id: Optional[str] = None
    enable_extract: Optional[bool] = None

    # policy-controlled generate clamp
    generate_max_new_tokens_cap: Optional[int] = None

    source_path: Optional[str] = None
    error: Optional[str] = None
    raw: Dict[str, Any] = Field(default_factory=dict)


class AdminReloadModels(BaseModel):
    default_model: str
    models: list[str]
    runtime_default_model: Optional[str] = None
    registry_kind: Optional[str] = None


class AdminReloadPolicy(BaseModel):
    snapshot_ok: bool
    model_id: Optional[str] = None
    enable_extract: Optional[bool] = None
    generate_max_new_tokens_cap: Optional[int] = None
    source_path: Optional[str] = None
    error: Optional[str] = None


class AdminReloadEffective(BaseModel):
    extract_enabled: bool


class AdminReloadResponse(BaseModel):
    ok: bool
    models: AdminReloadModels
    policy: AdminReloadPolicy
    effective: AdminReloadEffective


class AdminWriteGenerateSloResponse(BaseModel):
    ok: bool
    out_path: str
    payload: Dict[str, Any]


# ---- Runtime model ops ----


class AdminModelsStatusResponse(BaseModel):
    ok: bool
    model_error: Optional[str] = None
    model_load_mode: str
    model_loaded: bool
    loaded_model_id: Optional[str] = None
    runtime_default_model_id: Optional[str] = None
    models_config_loaded: bool
    registry_kind: Optional[str] = None
    registry: Optional[list[dict[str, Any]]] = None


class AdminProbeModelResponse(BaseModel):
    ok: bool
    model_id: str
    detail: Dict[str, Any] = Field(default_factory=dict)


class AdminSetDefaultModelRequest(BaseModel):
    model_id: str


class AdminSetDefaultModelResponse(BaseModel):
    ok: bool
    default_model: Optional[str] = None
    persisted: bool = False


# -------------------------------------------------------------------
# /v1/me/usage
# -------------------------------------------------------------------


@router.get("/v1/me/usage", response_model=MeUsageResponse)
async def get_my_usage(
    request: Request,
    api_key: ApiKey = Depends(get_api_key),
    session: AsyncSession = Depends(get_session),
):
    set_request_meta(request, route="/v1/me/usage", model_id="admin", cached=False)

    role_name = await telem_q.fetch_role_name(session, api_key.role_id)
    usage = await telem_q.get_me_usage(session, api_key_value=api_key.key, role_name=role_name)

    return MeUsageResponse(
        api_key=usage.api_key,
        role=usage.role,
        total_requests=usage.total_requests,
        first_request_at=usage.first_request_at,
        last_request_at=usage.last_request_at,
        total_prompt_tokens=usage.total_prompt_tokens,
        total_completion_tokens=usage.total_completion_tokens,
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
    set_request_meta(request, route="/v1/admin/usage", model_id="admin", cached=False)
    await ensure_admin(api_key, session)

    rows = await telem_q.get_admin_usage(session)
    return AdminUsageResponse(
        results=[
            AdminUsageRow(
                api_key=r.api_key,
                name=r.name,
                role=r.role,
                total_requests=r.total_requests,
                total_prompt_tokens=r.total_prompt_tokens,
                total_completion_tokens=r.total_completion_tokens,
                first_request_at=r.first_request_at,
                last_request_at=r.last_request_at,
            )
            for r in rows
        ]
    )


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
    set_request_meta(request, route="/v1/admin/keys", model_id="admin", cached=False)
    await ensure_admin(api_key, session)

    page = await telem_q.list_api_keys(session, limit=limit, offset=offset)
    return AdminApiKeyListResponse(
        results=[
            AdminApiKeyInfo(
                key_prefix=x.key_prefix,
                name=x.name,
                role=x.role,
                created_at=x.created_at,
                disabled=x.disabled,
            )
            for x in page.items
        ],
        total=page.total,
        limit=page.limit,
        offset=page.offset,
    )


# -------------------------------------------------------------------
# /v1/admin/logs
# -------------------------------------------------------------------


@router.get("/v1/admin/logs", response_model=AdminLogsPage)
async def list_inference_logs(
    request: Request,
    api_key: ApiKey = Depends(get_api_key),
    session: AsyncSession = Depends(get_session),
    model_id: Optional[str] = Query(default=None, description="Filter by model_id"),
    key: Optional[str] = Query(default=None, alias="api_key", description="Filter by API key value"),
    route: Optional[str] = Query(default=None, description="Filter by route, e.g. /v1/generate"),
    from_ts: Optional[datetime] = Query(default=None, description="Filter logs created_at >= this timestamp (ISO8601)"),
    to_ts: Optional[datetime] = Query(default=None, description="Filter logs created_at <= this timestamp (ISO8601)"),
    limit: int = Query(default=50, ge=1, le=200, description="Max number of rows to return"),
    offset: int = Query(default=0, ge=0, description="Offset for pagination"),
):
    set_request_meta(request, route="/v1/admin/logs", model_id="admin", cached=False)
    await ensure_admin(api_key, session)

    page = await telem_q.list_inference_logs(
        session,
        model_id=model_id,
        api_key_value=key,
        route=route,
        from_ts=from_ts,
        to_ts=to_ts,
        limit=limit,
        offset=offset,
    )

    items = [AdminLogEntry.model_validate(row) for row in page.items]
    return AdminLogsPage(total=page.total, limit=page.limit, offset=page.offset, items=items)


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
    set_request_meta(request, route="/v1/admin/stats", model_id="admin", cached=False)
    await ensure_admin(api_key, session)

    stats = await telem_q.get_admin_stats(session, window_days=window_days)

    return AdminStatsResponse(
        window_days=stats.window_days,
        since=stats.since,
        total_requests=stats.total_requests,
        total_prompt_tokens=stats.total_prompt_tokens,
        total_completion_tokens=stats.total_completion_tokens,
        avg_latency_ms=stats.avg_latency_ms,
        per_model=[
            AdminModelStats(
                model_id=m.model_id,
                total_requests=m.total_requests,
                total_prompt_tokens=m.total_prompt_tokens,
                total_completion_tokens=m.total_completion_tokens,
                avg_latency_ms=m.avg_latency_ms,
            )
            for m in stats.per_model
        ],
    )


# -------------------------------------------------------------------
# /v1/admin/reports/summary
# -------------------------------------------------------------------


@router.get("/v1/admin/reports/summary")
async def admin_report_summary(
    request: Request,
    api_key: ApiKey = Depends(get_api_key),
    session: AsyncSession = Depends(get_session),
    window_days: int = Query(30, ge=1, le=365),
    format: str = Query("text", pattern="^(text|json|md)$"),
):
    set_request_meta(request, route="/v1/admin/reports/summary", model_id="admin", cached=False)
    await ensure_admin(api_key, session)

    stats = await telem_q.get_admin_stats(session, window_days=window_days)

    stats_payload: Dict[str, Any] = {
        "window_days": stats.window_days,
        "since": stats.since,
        "total_requests": stats.total_requests,
        "total_prompt_tokens": stats.total_prompt_tokens,
        "total_completion_tokens": stats.total_completion_tokens,
        "avg_latency_ms": stats.avg_latency_ms,
    }
    per_model = [
        {
            "model_id": m.model_id,
            "total_requests": m.total_requests,
            "total_prompt_tokens": m.total_prompt_tokens,
            "total_completion_tokens": m.total_completion_tokens,
            "avg_latency_ms": m.avg_latency_ms,
        }
        for m in stats.per_model
    ]

    return report_w.render_admin_summary(stats_payload=stats_payload, per_model=per_model, fmt=format)


# -------------------------------------------------------------------
# /v1/admin/models/status
# -------------------------------------------------------------------


@router.get("/v1/admin/models/status", response_model=AdminModelsStatusResponse)
async def admin_models_status(
    request: Request,
    api_key: ApiKey = Depends(get_api_key),
    session: AsyncSession = Depends(get_session),
):
    set_request_meta(request, route="/v1/admin/models/status", model_id="admin", cached=False)
    await ensure_admin(api_key, session)

    loader = get_loader(request)
    snap = await loader.status()

    return AdminModelsStatusResponse(
        ok=True,
        model_error=cast(Optional[str], snap.get("model_error")),
        model_load_mode=str(snap.get("model_load_mode") or "unknown"),
        model_loaded=bool(snap.get("model_loaded", False)),
        loaded_model_id=cast(Optional[str], snap.get("loaded_model_id")),
        runtime_default_model_id=cast(Optional[str], snap.get("runtime_default_model_id")),
        models_config_loaded=bool(snap.get("models_config_loaded", False)),
        registry_kind=cast(Optional[str], snap.get("registry_kind")),
        registry=cast(Optional[list[dict[str, Any]]], snap.get("registry")),
    )


# -------------------------------------------------------------------
# /v1/admin/models/probe
# -------------------------------------------------------------------


@router.get("/v1/admin/models/probe", response_model=AdminProbeModelResponse)
async def admin_models_probe(
    request: Request,
    api_key: ApiKey = Depends(get_api_key),
    session: AsyncSession = Depends(get_session),
    model_id: str = Query(..., description="Model id to probe"),
):
    """
    Cheap probe (does NOT load weights). Useful for external backends.
    """
    set_request_meta(request, route="/v1/admin/models/probe", model_id="admin", cached=False)
    await ensure_admin(api_key, session)

    loader = get_loader(request)
    res = await loader.probe_model(model_id)

    return AdminProbeModelResponse(ok=bool(res.ok), model_id=res.model_id, detail=res.detail or {})


# -------------------------------------------------------------------
# /v1/admin/models/default (set/clear runtime default)
# -------------------------------------------------------------------


@router.post("/v1/admin/models/default", response_model=AdminSetDefaultModelResponse)
async def admin_models_set_default(
    request: Request,
    body: AdminSetDefaultModelRequest,
    api_key: ApiKey = Depends(get_api_key),
    session: AsyncSession = Depends(get_session),
):
    """
    Runtime-only default model override. Does not modify models.yaml.
    """
    set_request_meta(request, route="/v1/admin/models/default", model_id="admin", cached=False)
    await ensure_admin(api_key, session)

    loader = get_loader(request)
    info = await loader.set_default_model(body.model_id)

    return AdminSetDefaultModelResponse(
        ok=True,
        default_model=cast(Optional[str], info.get("default_model")),
        persisted=bool(info.get("persisted", False)),
    )


@router.post("/v1/admin/models/default/clear", response_model=AdminSetDefaultModelResponse)
async def admin_models_clear_default(
    request: Request,
    api_key: ApiKey = Depends(get_api_key),
    session: AsyncSession = Depends(get_session),
):
    set_request_meta(request, route="/v1/admin/models/default/clear", model_id="admin", cached=False)
    await ensure_admin(api_key, session)

    loader = get_loader(request)
    info = await loader.clear_runtime_default()

    return AdminSetDefaultModelResponse(
        ok=True,
        default_model=cast(Optional[str], info.get("default_model")),
        persisted=bool(info.get("persisted", False)),
    )


# -------------------------------------------------------------------
# /v1/admin/models/load  (refactored to RuntimeModelLoader)
# -------------------------------------------------------------------


@router.post("/v1/admin/models/load", response_model=AdminLoadModelResponse)
async def admin_load_model(
    request: Request,
    body: AdminLoadModelRequest,
    api_key: ApiKey = Depends(get_api_key),
    session: AsyncSession = Depends(get_session),
    force: bool = Query(False, description="Force reload (transformers only); external backends remain no-op"),
):
    """
    Explicit weight-load boundary.

    IMPORTANT:
      - This is the ONLY admin endpoint that should trigger weight loading.
      - For llama-server style external backends, ensure_loaded() should be cheap/no-op.
      - For transformers, this will typically allocate weights and can be slow.

    Semantics:
      - app.state.model_loaded means "transformers weights loaded in-process"
      - llama/remote must not flip model_loaded=True
    """
    set_request_meta(request, route="/v1/admin/models/load", model_id="admin", cached=False)
    await ensure_admin(api_key, session)

    loader = get_loader(request)

    # Validate against allowed model IDs (settings-based), if provided.
    if body.model_id:
        s = settings_from_request(request)
        allowed = allowed_model_ids_from_settings(s)
        if allowed and body.model_id not in allowed:
            raise AppError(
                code="model_not_allowed",
                message=f"Model '{body.model_id}' not allowed.",
                status_code=status.HTTP_400_BAD_REQUEST,
                extra={"allowed": allowed},
            )

        res = await loader.load_model(body.model_id, force=bool(force))
    else:
        res = await loader.load_default(force=bool(force))

    return AdminLoadModelResponse(
        ok=True,
        loaded=bool(res.loaded),
        model_id=str(res.model_id),
        load_mode=str(res.load_mode),
        detail=res.detail or {},
    )


# -------------------------------------------------------------------
# /v1/admin/policy (inspect/reload)
# -------------------------------------------------------------------


@router.get("/v1/admin/policy", response_model=AdminPolicySnapshotResponse)
async def admin_get_policy_snapshot(
    request: Request,
    api_key: ApiKey = Depends(get_api_key),
    session: AsyncSession = Depends(get_session),
):
    set_request_meta(request, route="/v1/admin/policy", model_id="admin", cached=False)
    await ensure_admin(api_key, session)

    snap = get_policy_snapshot(request)
    return AdminPolicySnapshotResponse(
        ok=bool(getattr(snap, "ok", False)),
        model_id=getattr(snap, "model_id", None),
        enable_extract=getattr(snap, "enable_extract", None),
        generate_max_new_tokens_cap=snapshot_generate_cap(snap),
        source_path=getattr(snap, "source_path", None),
        error=getattr(snap, "error", None),
        raw=getattr(snap, "raw", None) or {},
    )


@router.post("/v1/admin/policy/reload", response_model=AdminPolicySnapshotResponse)
async def admin_reload_policy_snapshot(
    request: Request,
    api_key: ApiKey = Depends(get_api_key),
    session: AsyncSession = Depends(get_session),
):
    set_request_meta(request, route="/v1/admin/policy/reload", model_id="admin", cached=False)
    await ensure_admin(api_key, session)

    snap = reload_policy_snapshot(request)
    return AdminPolicySnapshotResponse(
        ok=bool(getattr(snap, "ok", False)),
        model_id=getattr(snap, "model_id", None),
        enable_extract=getattr(snap, "enable_extract", None),
        generate_max_new_tokens_cap=snapshot_generate_cap(snap),
        source_path=getattr(snap, "source_path", None),
        error=getattr(snap, "error", None),
        raw=getattr(snap, "raw", None) or {},
    )


# -------------------------------------------------------------------
# /v1/admin/slo/generate/write
# -------------------------------------------------------------------


@router.post("/v1/admin/slo/generate/write", response_model=AdminWriteGenerateSloResponse)
async def admin_write_generate_slo(
    request: Request,
    api_key: ApiKey = Depends(get_api_key),
    session: AsyncSession = Depends(get_session),
    window_seconds: int = Query(300, ge=30, le=86400),
    route: Optional[str] = Query(default=None, description="Optional single route override (default: /v1/generate)"),
    model_id: Optional[str] = Query(default=None, description="Optional filter by model_id"),
    out_path: Optional[str] = Query(default=None, description="Optional artifact path override"),
):
    """
    Deterministic “button” for demo + tooling:
      - compute SLO snapshot from DB logs
      - write runtime_generate_slo_v1 artifact to disk
    """
    set_request_meta(request, route="/v1/admin/slo/generate/write", model_id="admin", cached=False)
    await ensure_admin(api_key, session)

    routes = [route] if isinstance(route, str) and route.strip() else ["/v1/generate", "/v1/generate/batch"]

    res = await write_generate_slo_artifact(
        session,
        window_seconds=window_seconds,
        routes=routes,
        model_id=model_id,
        out_path=out_path,
    )

    return AdminWriteGenerateSloResponse(ok=bool(res.ok), out_path=res.out_path, payload=res.payload)


# -------------------------------------------------------------------
# /v1/admin/reload  (refactored to RuntimeModelLoader)
# -------------------------------------------------------------------


@router.post("/v1/admin/reload", response_model=AdminReloadResponse)
async def admin_reload_runtime(
    request: Request,
    api_key: ApiKey = Depends(get_api_key),
    session: AsyncSession = Depends(get_session),
):
    """
    Deterministic reload boundary for server-side state.
    """
    set_request_meta(request, route="/v1/admin/reload", model_id="admin", cached=False)
    await ensure_admin(api_key, session)

    loader = get_loader(request)

    payload, _snap = await reload_runtime_state(request=request, loader=loader)

    return AdminReloadResponse(
        ok=True,
        models=AdminReloadModels(**payload["models"]),
        policy=AdminReloadPolicy(**payload["policy"]),
        effective=AdminReloadEffective(**payload["effective"]),
    )