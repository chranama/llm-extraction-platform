# server/src/llm_server/services/api_deps/extract/stage.py
from __future__ import annotations

from fastapi import Request, status

from llm_server.core.errors import AppError


def set_stage(request: Request, stage: str) -> None:
    try:
        request.state.error_stage = stage
    except Exception:
        pass


def failure_stage_for_app_error(e: AppError, *, is_repair: bool) -> str | None:
    if getattr(e, "status_code", None) != status.HTTP_422_UNPROCESSABLE_CONTENT:
        return None
    if e.code == "invalid_json":
        return "repair_parse" if is_repair else "parse"
    if e.code == "schema_validation_failed":
        return "repair_validate" if is_repair else "validate"
    if e.code == "possible_truncation":
        return "repair_truncation" if is_repair else "truncation"
    return "repair_validate" if is_repair else "validate"