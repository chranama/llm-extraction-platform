# server/src/llm_server/services/limits/early_reject_middleware.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from llm_server.core.errors import AppError
from llm_server.services.limits.generate_gating import get_generate_gate


@dataclass(frozen=True)
class EarlyRejectConfig:
    enabled: bool = True
    reject_queue_depth_gte: int = 0
    reject_in_flight_gte: int = 0
    routes: tuple[str, ...] = ("/v1/generate", "/v1/generate/batch")


def _truthy(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    return str(x).strip().lower() in {"1", "true", "yes", "y", "on"}


def _as_int(x: Any, default: int) -> int:
    if x is None or isinstance(x, bool):
        return default
    try:
        return int(x)
    except Exception:
        return default


def _get_attr_or_key(obj: Any, key: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _settings_fingerprint(app: Any) -> tuple[int, int]:
    """
    Best-effort fingerprint to invalidate cached config when settings change.
    (settings_id, limits_id)
    """
    s = getattr(getattr(app, "state", None), "settings", None)
    limits = _get_attr_or_key(s, "limits") if s is not None else None
    return (id(s), id(limits))


def _cfg_from_app(app: Any) -> EarlyRejectConfig:
    state = getattr(app, "state", None)

    # Optional micro-cache with invalidation
    cached = getattr(state, "generate_early_reject_cfg", None)
    cached_fp = getattr(state, "generate_early_reject_cfg_fp", None)
    fp = _settings_fingerprint(app)

    if isinstance(cached, EarlyRejectConfig) and cached_fp == fp:
        return cached

    s = getattr(state, "settings", None)
    if s is None:
        cfg = EarlyRejectConfig()
        try:
            state.generate_early_reject_cfg = cfg
            state.generate_early_reject_cfg_fp = fp
        except Exception:
            pass
        return cfg

    limits = _get_attr_or_key(s, "limits")
    er = _get_attr_or_key(limits, "generate_early_reject")

    if not isinstance(er, dict):
        cfg = EarlyRejectConfig()
        try:
            state.generate_early_reject_cfg = cfg
            state.generate_early_reject_cfg_fp = fp
        except Exception:
            pass
        return cfg

    enabled_raw = er.get("enabled", True)
    rq_raw = er.get("reject_queue_depth_gte", 0)
    rf_raw = er.get("reject_in_flight_gte", 0)
    routes_raw = er.get("routes", ["/v1/generate", "/v1/generate/batch"])

    enabled = _truthy(enabled_raw) if enabled_raw is not None else True
    rq = max(0, _as_int(rq_raw, 0))
    rf = max(0, _as_int(rf_raw, 0))

    routes: list[str] = []
    if isinstance(routes_raw, str):
        routes_raw = [routes_raw]
    if isinstance(routes_raw, (list, tuple)):
        for r in routes_raw:
            if isinstance(r, str) and r.strip():
                routes.append(r.strip())

    cfg = EarlyRejectConfig(
        enabled=bool(enabled),
        reject_queue_depth_gte=int(rq),
        reject_in_flight_gte=int(rf),
        routes=tuple(routes) if routes else ("/v1/generate", "/v1/generate/batch"),
    )

    try:
        state.generate_early_reject_cfg = cfg
        state.generate_early_reject_cfg_fp = fp
    except Exception:
        pass

    return cfg


class EarlyRejectGenerateMiddleware(BaseHTTPMiddleware):
    """
    Fast-path overload protection: reject before doing expensive work.

    This does NOT acquire gate tokens.
    It is an approximate "traffic cop" that prevents the server from
    spending time on requests that will almost certainly timeout anyway.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        cfg = _cfg_from_app(request.app)
        if not cfg.enabled:
            return await call_next(request)

        path = request.url.path
        if path not in cfg.routes:
            return await call_next(request)

        gate = get_generate_gate()
        snap = gate.snapshot()

        # If gate itself is disabled, keep consistent behavior (reject).
        if not snap.enabled:
            raise AppError(
                code="generate_overloaded",
                message="Generate is overloaded. Try again later.",
                status_code=429,
                extra={"reason": "disabled"},
            )

        if cfg.reject_in_flight_gte > 0 and snap.in_flight_estimate >= cfg.reject_in_flight_gte:
            raise AppError(
                code="generate_overloaded",
                message="Generate is overloaded. Try again later.",
                status_code=429,
                extra={
                    "reason": "in_flight_high",
                    "in_flight_estimate": snap.in_flight_estimate,
                    "queue_depth_estimate": snap.queue_depth_estimate,
                    "max_concurrent": snap.max_concurrent,
                    "max_queue": snap.max_queue,
                },
            )

        if cfg.reject_queue_depth_gte > 0 and snap.queue_depth_estimate >= cfg.reject_queue_depth_gte:
            raise AppError(
                code="generate_overloaded",
                message="Generate is overloaded. Try again later.",
                status_code=429,
                extra={
                    "reason": "queue_full",
                    "in_flight_estimate": snap.in_flight_estimate,
                    "queue_depth_estimate": snap.queue_depth_estimate,
                    "max_concurrent": snap.max_concurrent,
                    "max_queue": snap.max_queue,
                },
            )

        return await call_next(request)