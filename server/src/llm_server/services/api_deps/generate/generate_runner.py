# server/src/llm_server/services/generate/generate_runner.py
from __future__ import annotations

from typing import Any

import anyio


def _extract_usage_dict(usage_obj: Any) -> dict[str, Any] | None:
    """
    Best-effort normalization of a backend "usage" object to a dict:
      {"prompt_tokens": int|None, "completion_tokens": int|None, "total_tokens": int|None}
    """
    if usage_obj is None:
        return None
    try:
        pt = getattr(usage_obj, "prompt_tokens", None)
        ct = getattr(usage_obj, "completion_tokens", None)
        tt = getattr(usage_obj, "total_tokens", None)
        return {
            "prompt_tokens": int(pt) if isinstance(pt, int) else None,
            "completion_tokens": int(ct) if isinstance(ct, int) else None,
            "total_tokens": int(tt) if isinstance(tt, int) else None,
        }
    except Exception:
        return None


async def run_generate_rich_offloop(model: Any, **kwargs: Any) -> tuple[str, dict[str, Any] | None]:
    """
    Run generation off the event loop.

    Contract:
      - Prefer model.generate_rich(**kwargs) -> object with .text and optional .usage
      - Else fallback to model.generate(**kwargs) -> string-ish output
      - Always returns (text, usage_dict_or_none)
    """
    def _run() -> tuple[str, dict[str, Any] | None]:
        gen_rich = getattr(model, "generate_rich", None)
        if callable(gen_rich):
            r = gen_rich(**kwargs)
            text = str(getattr(r, "text", "") or "")
            usage_dict = _extract_usage_dict(getattr(r, "usage", None))
            return text, usage_dict

        gen = getattr(model, "generate", None)
        if not callable(gen):
            return "", None

        out = gen(**kwargs)
        return (out if isinstance(out, str) else str(out)), None

    return await anyio.to_thread.run_sync(_run)