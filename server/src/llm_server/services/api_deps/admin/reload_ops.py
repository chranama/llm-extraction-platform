# server/src/llm_server/services/api_deps/admin/reload_ops.py
from __future__ import annotations

import logging
from typing import Any, Optional, Tuple, cast

from fastapi import Request

from llm_server.core.config import get_settings
from llm_server.io.policy_decisions import reload_policy_snapshot
from llm_server.services.api_deps.core.models_config import clear_models_config_cache
from llm_server.services.api_deps.enforcement.capabilities import effective_capabilities
from llm_server.services.llm_runtime.llm_loader import RuntimeModelLoader

from llm_server.services.api_deps.admin.models_ops import summarize_registry, runtime_default_model_id_from_app
from llm_server.services.api_deps.core.policy_snapshot import snapshot_generate_cap

logger = logging.getLogger("llm_server.api_deps.admin.reload_ops")


async def reload_runtime_state(
    *,
    request: Request,
    loader: RuntimeModelLoader,
) -> Tuple[dict[str, Any], Any]:
    """
    Deterministic reload boundary for server-side state:

      - clear models config cache (deps layer)
      - refresh models_config (re-parse models.yaml)
      - rebuild llm registry object (clients/backends) WITHOUT loading weights
      - clear + reload policy snapshot from disk
      - return merged summary that matches runtime gating

    Returns:
      (payload_models_policy_effective, snap_obj)
    """
    s = get_settings()
    app = request.app

    # 1) Clear deps-level cache(s) so request-time reads see new models.yaml
    try:
        clear_models_config_cache()
    except Exception as e:
        logger.warning("clear_models_config_cache failed: %r", e)

    # 2) Clear policy snapshot cache explicitly (so reload is a true boundary)
    try:
        app.state.policy_snapshot = None
    except Exception:
        pass

    # 3) Clear model readiness/error (registry will be rebuilt; weights remain unloaded)
    try:
        app.state.model_error = None
        app.state.model_loaded = False
        app.state.loaded_model_id = None
    except Exception:
        pass

    # 4) Refresh models.yaml parse + rebuild registry (no weights)
    try:
        await loader.refresh_models_config()
    except Exception as e:
        logger.warning("refresh_models_config failed (continuing): %r", e)

    try:
        llm = await loader.rebuild_llm_registry()
        app.state.llm = llm
    except Exception as e:
        app.state.model_error = repr(e)
        raise

    default_model, model_ids = summarize_registry(
        getattr(app.state, "llm", None),
        fallback_default=cast(str, getattr(s, "model_id", "")),
    )

    # 5) Reload policy snapshot from disk (overwrites app.state cache)
    snap = reload_policy_snapshot(request)

    # 6) Compute EFFECTIVE extract enabled using the same gating as /v1/extract
    extract_enabled = False
    try:
        caps = effective_capabilities(default_model, request=request)
        extract_enabled = bool(caps.get("extract", False))
    except Exception:
        extract_enabled = False

    out = {
        "models": {
            "default_model": default_model,
            "models": model_ids,
            "runtime_default_model": runtime_default_model_id_from_app(request),
            "registry_kind": type(getattr(app.state, "llm", None)).__name__ if getattr(app.state, "llm", None) is not None else None,
        },
        "policy": {
            "snapshot_ok": bool(getattr(snap, "ok", False)),
            "model_id": getattr(snap, "model_id", None),
            "enable_extract": getattr(snap, "enable_extract", None),
            "generate_max_new_tokens_cap": snapshot_generate_cap(snap),
            "source_path": getattr(snap, "source_path", None),
            "error": getattr(snap, "error", None),
        },
        "effective": {
            "extract_enabled": bool(extract_enabled),
        },
    }
    return out, snap