# server/src/llm_server/services/api_deps/enforcement/capabilities.py
from __future__ import annotations

from typing import Any, Dict, Literal, Optional, cast

from fastapi import Request, status

from llm_server.core.errors import AppError
from llm_server.io.policy_decisions import policy_capability_overrides
from llm_server.services.api_deps.core.models_config import models_config_from_request
from llm_server.services.api_deps.core.settings import settings_from_request
from llm_server.services.llm_runtime.llm_registry import MultiModelManager

Capability = Literal["generate", "extract"]
_CAP_KEYS: tuple[Capability, ...] = ("generate", "extract")


def deployment_capabilities(request: Request | None = None) -> Dict[str, bool]:
    s = settings_from_request(request)
    return {
        "generate": bool(getattr(s, "enable_generate", True)),
        "extract": bool(getattr(s, "enable_extract", True)),
    }


def _model_capabilities_from_models_yaml(model_id: str, *, request: Request | None = None) -> Optional[Dict[str, bool]]:
    cfg = models_config_from_request(request)

    defaults_caps = getattr(cfg, "defaults", {}).get("capabilities") if hasattr(cfg, "defaults") else None
    defaults_caps = cast(Optional[Dict[str, bool]], defaults_caps) if isinstance(defaults_caps, dict) else None

    spec_caps: Optional[Dict[str, bool]] = None
    for sp in getattr(cfg, "models", []) or []:
        if getattr(sp, "id", None) == model_id:
            if isinstance(getattr(sp, "capabilities", None), dict):
                spec_caps = dict(getattr(sp, "capabilities"))
            break

    if defaults_caps is None and spec_caps is None:
        return None

    out: Dict[str, bool] = {}
    if defaults_caps:
        for k in _CAP_KEYS:
            if k in defaults_caps:
                out[k] = bool(defaults_caps[k])
    if spec_caps:
        for k in _CAP_KEYS:
            if k in spec_caps:
                out[k] = bool(spec_caps[k])

    return out


def model_capabilities(model_id: str, *, request: Request | None = None) -> Optional[Dict[str, bool]]:
    base_caps: Optional[Dict[str, bool]] = None

    if request is not None:
        llm = getattr(request.app.state, "llm", None)
        if isinstance(llm, MultiModelManager):
            caps_meta = llm._meta.get(model_id, {}).get("capabilities", None)
            if caps_meta is None:
                base_caps = None
            elif isinstance(caps_meta, dict):
                out: Dict[str, bool] = {}
                for k in _CAP_KEYS:
                    if k in caps_meta:
                        out[k] = bool(caps_meta.get(k))
                base_caps = out or None
            elif isinstance(caps_meta, (list, tuple, set)):
                allowed = {str(x).strip().lower() for x in caps_meta if isinstance(x, str) and str(x).strip()}
                base_caps = {k: (k in allowed) for k in _CAP_KEYS}
            else:
                base_caps = None
        else:
            base_caps = _model_capabilities_from_models_yaml(model_id, request=request)
    else:
        base_caps = _model_capabilities_from_models_yaml(model_id, request=None)

    if request is not None:
        pol = policy_capability_overrides(model_id, request=request)
        if pol:
            merged: Dict[str, bool] = dict(base_caps or {})
            for k, v in pol.items():
                if k in _CAP_KEYS:
                    merged[k] = bool(v)
            return merged or None

    return base_caps


def effective_capabilities(model_id: str, *, request: Request | None = None) -> Dict[str, bool]:
    raw = model_capabilities(model_id, request=request)
    if raw is None:
        raw = {"generate": True, "extract": True}

    for k in _CAP_KEYS:
        raw.setdefault(k, True)

    dep = deployment_capabilities(request)
    return {k: bool(raw[k]) and bool(dep.get(k, True)) for k in _CAP_KEYS}


def require_capability(model_id: str, capability: Capability, *, request: Request | None = None) -> None:
    dep = deployment_capabilities(request)
    if not bool(dep.get(capability, True)):
        raise AppError(
            code="capability_disabled",
            message=f"{capability} is disabled in this deployment mode.",
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            extra={"capability": capability},
        )

    caps = model_capabilities(model_id, request=request)
    if caps is None:
        return

    supported = bool(caps.get(capability, True))
    if not supported:
        raise AppError(
            code="capability_not_supported",
            message=f"Model '{model_id}' does not support capability '{capability}'.",
            status_code=status.HTTP_400_BAD_REQUEST,
            extra={"model_id": model_id, "capability": capability, "model_capabilities": caps},
        )