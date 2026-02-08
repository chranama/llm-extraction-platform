# server/src/llm_server/api/admin.py
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, cast

from fastapi import APIRouter, Depends, Query, Request, status
from pydantic import BaseModel, ConfigDict
from sqlalchemy.ext.asyncio import AsyncSession

from llm_server.api.deps import clear_models_config_cache, effective_capabilities, get_api_key
from llm_server.core.config import get_settings
from llm_server.core.errors import AppError
from llm_server.db.models import ApiKey
from llm_server.db.session import get_session
from llm_server.io.policy_decisions import get_policy_snapshot, reload_policy_snapshot
from llm_server.io.runtime_generate_slo import write_generate_slo_artifact
from llm_server.reports import writer as report_w
from llm_server.services.inference import set_request_meta
from llm_server.services.llm import build_llm_from_settings
from llm_server.services.llm_registry import MultiModelManager
from llm_server.telemetry import queries as telem_q

logger = logging.getLogger("llm_server.api.admin")

router = APIRouter(tags=["admin"])

_MODEL_LOAD_LOCK = asyncio.Lock()


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
    model_id: Optional[str] = None


class AdminLoadModelResponse(BaseModel):
    ok: bool
    already_loaded: bool
    default_model: str
    models: list[str]


class AdminPolicySnapshotResponse(BaseModel):
    ok: bool
    model_id: Optional[str] = None
    enable_extract: Optional[bool] = None

    # NEW: policy-controlled generate clamp
    generate_max_new_tokens_cap: Optional[int] = None

    source_path: Optional[str] = None
    error: Optional[str] = None
    raw: Dict[str, Any] = {}


class AdminReloadModels(BaseModel):
    default_model: str
    models: list[str]


class AdminReloadPolicy(BaseModel):
    snapshot_ok: bool
    model_id: Optional[str] = None
    enable_extract: Optional[bool] = None

    # NEW: policy-controlled generate clamp
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


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


def _allowed_model_ids_from_settings(s) -> list[str]:
    allowed = getattr(s, "allowed_models", None)
    if isinstance(allowed, list) and allowed:
        return [str(x) for x in allowed if str(x).strip()]
    legacy = getattr(s, "all_model_ids", None) or []
    return [str(x) for x in legacy if str(x).strip()]


def _summarize_registry(llm_obj: Any, *, fallback_default: str) -> Tuple[str, list[str]]:
    if isinstance(llm_obj, MultiModelManager):
        return llm_obj.default_id, list(llm_obj.models.keys())

    if hasattr(llm_obj, "models") and hasattr(llm_obj, "default_id"):
        default_model = str(getattr(llm_obj, "default_id", "") or "") or fallback_default
        try:
            models_map = getattr(llm_obj, "models", {}) or {}
            model_ids = list(models_map.keys()) if isinstance(models_map, dict) else []
        except Exception:
            model_ids = []
        if not model_ids and default_model:
            model_ids = [default_model]
        return default_model, model_ids

    default_model = str(getattr(llm_obj, "model_id", "") or "") or fallback_default
    return default_model, [default_model] if default_model else []


def _snapshot_generate_cap(snap: Any) -> Optional[int]:
    """
    Back/forward compatible accessor:
      - prefer attribute on snapshot (future-proof)
      - fall back to raw dict keys
    """
    cap = getattr(snap, "generate_max_new_tokens_cap", None)
    if isinstance(cap, int):
        return cap
    raw = getattr(snap, "raw", None)
    if isinstance(raw, dict):
        v = raw.get("generate_max_new_tokens_cap")
        if isinstance(v, int):
            return v
    return None


async def _ensure_admin(api_key: ApiKey, session: AsyncSession) -> None:
    """
    Reload ApiKey with its Role in the current async session and enforce admin role.
    """
    db_key = await telem_q.reload_key_with_role(session, api_key_id=api_key.id)
    role_name = db_key.role.name if db_key and db_key.role else None
    if role_name != "admin":
        raise AppError(code="forbidden", message="Admin privileges required", status_code=status.HTTP_403_FORBIDDEN)
    
def _pick_model_to_load(*, requested: Optional[str], fallback_default: str) -> str:
    mid = (requested or "").strip()
    return mid or (fallback_default or "").strip()


def _force_load_llm(llm: Any, *, model_id: str) -> bool:
    """
    Best-effort load trigger across different registry implementations.

    Returns:
      True if we believe weights are loaded and the model is ready.
      False if we could not force-load (e.g., registry lacks load methods).
    Raises:
      Exception if the load method raises.
    """
    # 1) MultiModelManager-style APIs (try common method names)
    for meth in (
        "ensure_model_loaded",
        "load_model",
        "load",
        "ensure_loaded",  # may accept model_id or be no-arg
    ):
        fn = getattr(llm, meth, None)
        if callable(fn):
            try:
                # Prefer model-specific signature if it accepts one
                return bool(fn(model_id))  # type: ignore[misc]
            except TypeError:
                # Fallback to no-arg signature
                fn()  # type: ignore[misc]
                return True

    # 2) Some registries keep per-model objects with ensure_loaded
    models = getattr(llm, "models", None)
    if isinstance(models, dict) and model_id in models:
        mobj = models[model_id]
        fn = getattr(mobj, "ensure_loaded", None)
        if callable(fn):
            fn()
            return True

    # 3) Nothing matched
    return False


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
    await _ensure_admin(api_key, session)

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
    await _ensure_admin(api_key, session)

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
    await _ensure_admin(api_key, session)

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
    await _ensure_admin(api_key, session)

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
    await _ensure_admin(api_key, session)

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
# /v1/admin/models/load
# -------------------------------------------------------------------


@router.post("/v1/admin/models/load", response_model=AdminLoadModelResponse)
async def admin_load_model(
    request: Request,
    body: AdminLoadModelRequest,
    api_key: ApiKey = Depends(get_api_key),
    session: AsyncSession = Depends(get_session),
):
    set_request_meta(request, route="/v1/admin/models/load", model_id="admin", cached=False)
    await _ensure_admin(api_key, session)

    s = get_settings()
    app = request.app

    async with _MODEL_LOAD_LOCK:
        # If we already have a healthy loaded model, short-circuit
        existing = getattr(app.state, "llm", None)
        model_loaded = bool(getattr(app.state, "model_loaded", False))
        model_error = getattr(app.state, "model_error", None)

        if existing is not None and model_loaded and not model_error:
            default_model, model_ids = _summarize_registry(
                existing,
                fallback_default=cast(str, getattr(s, "model_id", "")),
            )
            return AdminLoadModelResponse(ok=True, already_loaded=True, default_model=default_model, models=model_ids)

        # If a model_id was requested, validate against allowed models
        if body.model_id:
            allowed = _allowed_model_ids_from_settings(s)
            if body.model_id not in allowed:
                raise AppError(
                    code="model_not_allowed",
                    message=f"Model '{body.model_id}' not allowed.",
                    status_code=status.HTTP_400_BAD_REQUEST,
                    extra={"allowed": allowed},
                )

            # Make requested model the default for this process
            s.model_id = body.model_id  # type: ignore[attr-defined]
            try:
                clear_models_config_cache()
            except Exception:
                pass

        # Reset runtime state
        app.state.model_error = None
        app.state.model_loaded = False
        app.state.llm = None

        # Build registry / llm object (should be cheap in lazy mode)
        try:
            llm = build_llm_from_settings()
            app.state.llm = llm
        except Exception as e:
            app.state.model_error = repr(e)
            app.state.model_loaded = False
            app.state.llm = None
            raise

        # Determine the model we will force-load
        fallback_default = cast(str, getattr(get_settings(), "model_id", "") or "")
        model_to_load = _pick_model_to_load(requested=body.model_id, fallback_default=fallback_default)

        # Force-load weights now (admin override to lazy)
        try:
            loaded = _force_load_llm(app.state.llm, model_id=model_to_load)
            app.state.model_loaded = bool(loaded)
        except Exception as e:
            app.state.model_error = repr(e)
            app.state.model_loaded = False
            app.state.llm = None
            raise

        default_model, model_ids = _summarize_registry(
            app.state.llm,
            fallback_default=fallback_default,
        )

        return AdminLoadModelResponse(
            ok=True,
            already_loaded=False,
            default_model=default_model,
            models=model_ids,
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
    await _ensure_admin(api_key, session)

    snap = get_policy_snapshot(request)
    return AdminPolicySnapshotResponse(
        ok=bool(getattr(snap, "ok", False)),
        model_id=getattr(snap, "model_id", None),
        enable_extract=getattr(snap, "enable_extract", None),
        generate_max_new_tokens_cap=_snapshot_generate_cap(snap),
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
    await _ensure_admin(api_key, session)

    snap = reload_policy_snapshot(request)
    return AdminPolicySnapshotResponse(
        ok=bool(getattr(snap, "ok", False)),
        model_id=getattr(snap, "model_id", None),
        enable_extract=getattr(snap, "enable_extract", None),
        generate_max_new_tokens_cap=_snapshot_generate_cap(snap),
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
    await _ensure_admin(api_key, session)

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
# /v1/admin/reload
# -------------------------------------------------------------------


@router.post("/v1/admin/reload", response_model=AdminReloadResponse)
async def admin_reload_runtime(
    request: Request,
    api_key: ApiKey = Depends(get_api_key),
    session: AsyncSession = Depends(get_session),
):
    """
    Deterministic reload boundary:
      - clear models config cache
      - rebuild llm registry object (respects lazy/off modes)
      - clear + reload policy snapshot from disk
      - return merged state summary that matches runtime gating

    Now includes policy-driven generate clamp visibility.
    """
    set_request_meta(request, route="/v1/admin/reload", model_id="admin", cached=False)
    await _ensure_admin(api_key, session)

    s = get_settings()
    app = request.app

    # 1) Clear config cache(s)
    try:
        clear_models_config_cache()
    except Exception as e:
        logger.warning("clear_models_config_cache failed: %r", e)

    # 2) Clear cached runtime objects so config changes take effect deterministically
    app.state.llm = None
    app.state.model_loaded = False
    app.state.model_error = None

    # 2b) Clear policy snapshot cache explicitly (so reload is a true boundary)
    try:
        app.state.policy_snapshot = None
    except Exception:
        pass

    # 3) Rebuild llm registry (should not warm weights in lazy/off modes)
    try:
        llm = build_llm_from_settings()
        app.state.llm = llm
    except Exception as e:
        app.state.model_error = repr(e)
        raise

    default_model, model_ids = _summarize_registry(
        app.state.llm,
        fallback_default=cast(str, getattr(s, "model_id", "")),
    )

    # 4) Reload policy snapshot from disk (overwrites app.state cache)
    snap = reload_policy_snapshot(request)

    # 5) Compute EFFECTIVE extract enabled using the same gating as /v1/extract
    extract_enabled = False
    try:
        caps = effective_capabilities(default_model, request=request)
        extract_enabled = bool(caps.get("extract", False))
    except Exception:
        extract_enabled = False

    return AdminReloadResponse(
        ok=True,
        models=AdminReloadModels(default_model=default_model, models=model_ids),
        policy=AdminReloadPolicy(
            snapshot_ok=bool(getattr(snap, "ok", False)),
            model_id=getattr(snap, "model_id", None),
            enable_extract=getattr(snap, "enable_extract", None),
            generate_max_new_tokens_cap=_snapshot_generate_cap(snap),
            source_path=getattr(snap, "source_path", None),
            error=getattr(snap, "error", None),
        ),
        effective=AdminReloadEffective(extract_enabled=extract_enabled),
    )