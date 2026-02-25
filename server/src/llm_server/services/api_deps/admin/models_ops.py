# server/src/llm_server/services/api_deps/admin/models_ops.py
from __future__ import annotations

from typing import Any, Optional, Tuple, cast

from fastapi import Request

from llm_server.services.api_deps.enforcement.model_ready import get_runtime_model_loader
from llm_server.services.llm_runtime.llm_registry import MultiModelManager
from llm_server.services.llm_runtime.llm_loader import RuntimeModelLoader


def allowed_model_ids_from_settings(s: Any) -> list[str]:
    allowed = getattr(s, "allowed_models", None)
    if isinstance(allowed, list) and allowed:
        return [str(x) for x in allowed if str(x).strip()]
    legacy = getattr(s, "all_model_ids", None) or []
    return [str(x) for x in legacy if str(x).strip()]


def summarize_registry(llm_obj: Any, *, fallback_default: str) -> Tuple[str, list[str]]:
    """
    Returns (default_model, list_of_models) for various registry shapes.
    """
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


def runtime_default_model_id_from_app(request: Request) -> Optional[str]:
    v = getattr(request.app.state, "runtime_default_model_id", None)
    if isinstance(v, str) and v.strip():
        return v.strip()
    return None


def get_loader(request: Request) -> RuntimeModelLoader:
    return cast(RuntimeModelLoader, get_runtime_model_loader(request))