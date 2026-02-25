# server/src/llm_server/services/api_deps/core/models_config.py
from __future__ import annotations

from functools import lru_cache
from typing import Any

from fastapi import Request

from llm_server.services.llm_runtime.llm_config import load_models_config


@lru_cache(maxsize=1)
def _cached_models_config() -> Any:
    return load_models_config()


def clear_models_config_cache() -> None:
    _cached_models_config.cache_clear()


def models_config_from_request(request: Request | None) -> Any:
    """
    Prefer request-scoped parsed config (app.state.models_config) when available.
    Falls back to a tiny process cache to avoid reparsing models.yaml repeatedly.
    """
    if request is not None:
        cfg = getattr(request.app.state, "models_config", None)
        if cfg is not None:
            return cfg
    return _cached_models_config()