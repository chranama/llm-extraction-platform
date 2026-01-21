# src/llm_server/api/capabilities.py
from __future__ import annotations

from fastapi import APIRouter

from llm_server.core.config import get_settings

router = APIRouter(prefix="/v1", tags=["capabilities"])


@router.get("/capabilities")
def get_capabilities():
    """
    Report which product capabilities are enabled in this deployment.

    This is intentionally simple and config-driven in Phase 1.
    In Phase 2, you may optionally layer in runtime probes to mark
    capabilities unavailable even if enabled by config.
    """
    s = get_settings()

    enable_generate = bool(getattr(s, "enable_generate", True))
    enable_extract = bool(getattr(s, "enable_extract", True))

    mode = "generate+extract" if (enable_generate and enable_extract) else "generate-only"

    return {
        "generate": enable_generate,
        "extract": enable_extract,
        "mode": mode,
    }