# server/src/llm_server/services/api_deps/core/settings.py
from __future__ import annotations

from typing import Any

from fastapi import Request

from llm_server.core.config import get_settings


def settings_from_request(request: Request | None) -> Any:
    """
    Prefer request-scoped settings (app.state.settings) when available.
    Falls back to global settings loader.
    """
    if request is not None:
        s = getattr(request.app.state, "settings", None)
        if s is not None:
            return s
    return get_settings()