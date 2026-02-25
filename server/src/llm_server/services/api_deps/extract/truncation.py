# server/src/llm_server/services/api_deps/extract/truncation.py
from __future__ import annotations

from fastapi import status

from llm_server.core.errors import AppError
from llm_server.services.api_deps.extract.constants import _JSON_BEGIN, _JSON_END


def maybe_raise_truncation_error(
    *,
    raw_output: str,
    effective_max_new_tokens: int | None,
    applied_cap: int | None,
    stage: str,
) -> None:
    # Only meaningful if a policy cap was actually applied.
    if not applied_cap:
        return
    if effective_max_new_tokens is None:
        return

    s = (raw_output or "").strip()
    if not s:
        return

    has_begin = _JSON_BEGIN in s
    has_end = _JSON_END in s
    brace_delta = s.count("{") - s.count("}")

    looks_truncated = False
    if has_begin and not has_end:
        looks_truncated = True
    if brace_delta > 0:
        looks_truncated = True

    if not looks_truncated:
        return

    raise AppError(
        code="possible_truncation",
        message="Model output appears truncated; max_new_tokens may be too low for extraction.",
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        extra={
            "stage": stage,
            "effective_max_new_tokens": int(effective_max_new_tokens),
            "applied_policy_cap": int(applied_cap) if applied_cap is not None else None,
            "has_json_begin": bool(has_begin),
            "has_json_end": bool(has_end),
            "brace_delta": int(brace_delta),
            "raw_preview": s[:500],
        },
    )