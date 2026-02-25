# server/src/llm_server/services/api_deps/core/policy_snapshot.py
from __future__ import annotations

from typing import Any, Dict, Optional


def snapshot_generate_cap(snap: Any) -> Optional[int]:
    """
    Best-effort extraction of generate_max_new_tokens_cap from a policy snapshot object.

    Supports:
      - snap.generate_max_new_tokens_cap (int)
      - snap.raw["generate_max_new_tokens_cap"] (int)
    """
    cap = getattr(snap, "generate_max_new_tokens_cap", None)
    if isinstance(cap, int):
        return cap

    raw = getattr(snap, "raw", None)
    if isinstance(raw, dict):
        v = raw.get("generate_max_new_tokens_cap")
        if isinstance(v, int):
            return v
    return None


def policy_snapshot_summary(snap: Any) -> Dict[str, Any]:
    """
    Normalize a policy snapshot object into a small stable dict for health/admin output.
    """
    def _get(name: str, default=None):
        try:
            return getattr(snap, name, default)
        except Exception:
            return default

    raw = _get("raw", None)
    if not isinstance(raw, dict):
        raw = {}

    return {
        "ok": bool(_get("ok", False)),
        "source_path": _get("source_path", None),
        "model_id": _get("model_id", None),
        "enable_extract": _get("enable_extract", None),
        "generate_max_new_tokens_cap": snapshot_generate_cap(snap),
        "error": _get("error", None),
        "raw": raw,
    }