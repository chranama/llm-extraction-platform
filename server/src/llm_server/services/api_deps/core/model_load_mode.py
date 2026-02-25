# server/src/llm_server/services/api_deps/core/model_load_mode.py
from __future__ import annotations

from fastapi import Request

from llm_server.services.api_deps.core.settings import settings_from_request


def _normalize_mode(v: str) -> str:
    m = v.strip().lower()
    return "eager" if m == "on" else m


def effective_model_load_mode_from_request(request: Request) -> str:
    """
    Effective model_load_mode resolution.

    Priority:
      1) app.state.model_load_mode (runtime override)
      2) settings.model_load_mode
      3) env heuristic: prod => eager, else lazy

    Normalizes:
      - "on" => "eager"
    """
    mode = getattr(request.app.state, "model_load_mode", None)
    if isinstance(mode, str) and mode.strip():
        return _normalize_mode(mode)

    s = settings_from_request(request)
    raw = getattr(s, "model_load_mode", None)
    if isinstance(raw, str) and raw.strip():
        return _normalize_mode(raw)

    env = str(getattr(s, "env", "dev")).strip().lower()
    return "eager" if env == "prod" else "lazy"