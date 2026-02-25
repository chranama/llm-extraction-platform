# server/src/llm_server/services/api_deps/core/cache_keys.py
from __future__ import annotations

import hashlib
import json
from typing import Any


def sha32(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def sha32_json(params: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(params, sort_keys=True).encode("utf-8")).hexdigest()[:32]


def fingerprint_pydantic(body: Any, *, exclude: set[str], exclude_none: bool = True) -> str:
    """
    Stable fingerprint for cache keys. Assumes Pydantic v2.
    """
    params = body.model_dump(exclude=exclude, exclude_none=exclude_none)
    return sha32_json(params)


def make_cache_redis_key(model_id: str, prompt_hash: str, params_fp: str) -> str:
    return f"llm:cache:{model_id}:{prompt_hash}:{params_fp}"


def make_extract_redis_key(model_id: str, prompt_hash: str, params_fp: str) -> str:
    return f"llm:extract:{model_id}:{prompt_hash}:{params_fp}"