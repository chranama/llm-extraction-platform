# server/src/llm_server/services/api_deps/routing/models.py
from __future__ import annotations

from typing import Any, Literal, Optional, Tuple, cast

from fastapi import Request

from llm_server.core.errors import AppError
from llm_server.services.api_deps.core.settings import settings_from_request
from llm_server.services.llm_runtime.llm_registry import MultiModelManager
from llm_server.services.llm_runtime.model_state import ModelStateStore

Capability = Literal["generate", "extract"]


def allowed_model_ids(*, request: Request | None = None) -> list[str]:
    s = settings_from_request(request)
    allowed = getattr(s, "allowed_models", None)
    if isinstance(allowed, list) and allowed:
        return [str(x) for x in allowed if str(x).strip()]
    legacy = getattr(s, "all_model_ids", None) or []
    return [str(x) for x in legacy if str(x).strip()]


def allowed_model_ids_from_settings(s: Any) -> list[str]:
    allowed = getattr(s, "allowed_models", None)
    if isinstance(allowed, list) and allowed:
        return [str(x) for x in allowed if str(x).strip()]
    legacy = getattr(s, "all_model_ids", None) or []
    return [str(x) for x in legacy if str(x).strip()]


def default_model_id_from_settings(*, request: Request | None = None) -> str:
    s = settings_from_request(request)
    mid = getattr(s, "model_id", None)
    if not isinstance(mid, str) or not mid.strip():
        return ""
    return mid.strip()


def runtime_default_model_id(request: Request | None) -> str | None:
    """
    Prefer structured ModelStateStore if available, else app.state.runtime_default_model_id.
    """
    if request is None:
        return None

    try:
        snap = ModelStateStore(request.app.state).snapshot()
        v = getattr(snap, "runtime_default_model_id", None)
        if isinstance(v, str) and v.strip():
            return v.strip()
    except Exception:
        pass

    v = getattr(request.app.state, "runtime_default_model_id", None)
    if isinstance(v, str) and v.strip():
        return v.strip()
    return None


def loaded_model_id(request: Request | None) -> str | None:
    """
    Best-effort current loaded model id (authoritative for single-backend deployments).

    Prefers structured ModelStateStore; falls back to legacy app.state.loaded_model_id.
    """
    if request is None:
        return None

    try:
        snap = ModelStateStore(request.app.state).snapshot()
        v = getattr(snap, "loaded_model_id", None)
        if isinstance(v, str) and v.strip():
            return v.strip()
    except Exception:
        pass

    v = getattr(request.app.state, "loaded_model_id", None)
    if isinstance(v, str) and v.strip():
        return v.strip()
    return None


def resolve_default_model_id_and_backend_obj(request: Request) -> tuple[Optional[str], Optional[str], Any]:
    """
    Determine the currently-selected effective model id + backend object without forcing loads.

    Multi-model (MultiModelManager) priority:
      1) runtime_default_model_id (if exists in registry)
      2) MultiModelManager.default_id (if exists in registry)
      3) fallback (default_id only, backend_obj None)

    Single-backend priority:
      1) loaded_model_id (from ModelStateStore) if present
      2) backend_obj.model_id
      3) None

    Rationale:
      - In single-backend mode, there is only one actually-loaded model. If that model
        can differ from the configured default, health/modelz should reflect reality.
      - This function must never trigger weight loading.
    """
    llm = getattr(request.app.state, "llm", None)

    # Multi-model registry
    if isinstance(llm, MultiModelManager):
        runtime_mid = runtime_default_model_id(request)
        if isinstance(runtime_mid, str) and runtime_mid and runtime_mid in llm:
            backend_obj = llm[runtime_mid]
            return runtime_mid, getattr(backend_obj, "backend_name", None), backend_obj

        default_id = getattr(llm, "default_id", None)
        if isinstance(default_id, str) and default_id and default_id in llm:
            backend_obj = llm[default_id]
            return default_id, getattr(backend_obj, "backend_name", None), backend_obj

        # fallback: empty registry or mismatch
        return cast(Optional[str], getattr(llm, "default_id", None)), None, None

    # Single backend (or None)
    backend_obj = llm
    backend_name = getattr(backend_obj, "backend_name", None) if backend_obj is not None else None

    # Prefer authoritative loaded model id if present
    effective_mid = loaded_model_id(request)
    if not effective_mid:
        mid = getattr(backend_obj, "model_id", None) if backend_obj is not None else None
        effective_mid = str(mid).strip() if isinstance(mid, str) and str(mid).strip() else None

    return effective_mid, (str(backend_name) if isinstance(backend_name, str) else None), backend_obj


def model_flags_from_app_state(request: Request) -> tuple[bool, Optional[str], Optional[str], Optional[str]]:
    """
    Returns:
      (model_loaded, model_error, loaded_model_id, runtime_default_model_id)

    Prefers structured ModelStateStore, falls back to legacy app.state fields.
    """
    try:
        snap = ModelStateStore(request.app.state).snapshot()
        return (
            bool(getattr(snap, "model_loaded", False)),
            getattr(snap, "model_error", None),
            getattr(snap, "loaded_model_id", None),
            getattr(snap, "runtime_default_model_id", None),
        )
    except Exception:
        loaded_mid = getattr(request.app.state, "loaded_model_id", None)
        loaded_mid = loaded_mid.strip() if isinstance(loaded_mid, str) and loaded_mid.strip() else None

        model_error = getattr(request.app.state, "model_error", None)
        model_error = model_error.strip() if isinstance(model_error, str) and model_error.strip() else None

        return (
            bool(getattr(request.app.state, "model_loaded", False)),
            model_error,
            loaded_mid,
            runtime_default_model_id(request),
        )


def settings_readiness_mode(request: Request) -> str:
    """
    Settings-level readiness mode fallback.
    Allowed: off | probe | generate
    Default: generate
    """
    s = settings_from_request(request)
    v = getattr(s, "model_readiness_mode", None)
    if isinstance(v, str) and v.strip():
        vv = v.strip().lower()
        return vv if vv in ("off", "probe", "generate") else "generate"
    return "generate"


def per_model_readiness_mode(request: Request, model_id: Optional[str]) -> str:
    """
    Precedence:
      1) Per-model readiness_mode from registry meta (if MultiModelManager and present)
      2) Settings.model_readiness_mode fallback
    """
    fallback = settings_readiness_mode(request)

    if not model_id:
        return fallback

    llm = getattr(request.app.state, "llm", None)
    if isinstance(llm, MultiModelManager):
        try:
            meta = llm._meta.get(model_id, {}) if hasattr(llm, "_meta") else {}
        except Exception:
            meta = {}
        rm = meta.get("readiness_mode", None) if isinstance(meta, dict) else None
        if isinstance(rm, str) and rm.strip():
            v = rm.strip().lower()
            if v in ("off", "probe", "generate"):
                return v

    return fallback


def resolve_model(
    llm: Any,
    model_override: str | None,
    *,
    capability: Capability | None = None,
    request: Request | None = None,
) -> tuple[str, Any]:
    """
    Resolve (model_id, backend_obj) for either a MultiModelManager or a single backend.

    Enforces settings allow-list (allowed_models / all_model_ids) but does NOT enforce capability support.
    """
    allowed = allowed_model_ids(request=request)

    if isinstance(llm, MultiModelManager):
        if model_override is not None:
            model_id = model_override
            if model_id not in llm:
                raise AppError(
                    code="model_missing",
                    message=f"Model '{model_id}' not found in LLM registry",
                    status_code=500,
                    extra={"available": llm.list_models(), "default_id": llm.default_id},
                )
        else:
            model_id = runtime_default_model_id(request) or llm.default_id

            if (model_id == llm.default_id) and capability:
                fn = getattr(llm, "default_for_capability", None)
                if callable(fn):
                    try:
                        model_id = str(fn(capability))
                    except Exception:
                        model_id = llm.default_id

        if allowed and model_id not in allowed:
            raise AppError(
                code="model_not_allowed",
                message=f"Model '{model_id}' not allowed.",
                status_code=400,
                extra={"allowed": allowed},
            )

        return model_id, llm[model_id]

    if isinstance(llm, dict):
        model_id = model_override or default_model_id_from_settings(request=request) or next(iter(llm.keys()), "")
        if not model_id:
            raise AppError(code="model_config_invalid", message="No model configured", status_code=500)

        if model_override is not None and allowed and model_id not in allowed:
            raise AppError(
                code="model_not_allowed",
                message=f"Model '{model_id}' not allowed.",
                status_code=400,
                extra={"allowed": allowed},
            )

        if model_id not in llm:
            raise AppError(code="model_missing", message=f"Model '{model_id}' not found in LLM registry", status_code=500)

        return model_id, llm[model_id]

    # Single backend: for requestless contexts fall back to default_model_id_from_settings;
    # when request is available, prefer the currently loaded model id for fidelity.
    if request is not None and model_override is None:
        eff_mid = loaded_model_id(request)
        if eff_mid:
            model_id = eff_mid
        else:
            model_id = default_model_id_from_settings(request=request)
    else:
        model_id = model_override or default_model_id_from_settings(request=request)

    if model_override is not None and allowed and model_id not in allowed:
        raise AppError(
            code="model_not_allowed",
            message=f"Model '{model_id}' not allowed.",
            status_code=400,
            extra={"allowed": allowed},
        )

    return model_id or "default", llm