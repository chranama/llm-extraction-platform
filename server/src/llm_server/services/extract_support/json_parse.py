from __future__ import annotations

import html
import json
import re
from typing import Any, Optional

from fastapi import status

from llm_server.core.errors import AppError
from llm_server.core.validation import (
    DependencyMissingError,
    JSONSchemaValidationError,
    validate_jsonschema,
)
from llm_server.services.extract_support.constants import _JSON_BEGIN, _JSON_END


def strip_wrapping_code_fences(s: str) -> str:
    s = s.strip()
    if not s.startswith("```"):
        return s
    s = re.sub(r"^\s*```[a-zA-Z0-9]*\s*\n?", "", s)
    s = re.sub(r"\n?\s*```\s*$", "", s)
    return s.strip()


def iter_json_objects(raw: str) -> list[dict[str, Any]]:
    s = strip_wrapping_code_fences(raw)
    dec = json.JSONDecoder()

    objs: list[dict[str, Any]] = []
    i = 0
    n = len(s)

    while i < n:
        j = s.find("{", i)
        if j == -1:
            break
        try:
            obj, end = dec.raw_decode(s[j:])
            if isinstance(obj, dict):
                objs.append(obj)
            i = j + max(end, 1)
        except Exception:
            i = j + 1

    return objs


def _schema_tag_object_candidate(schema: dict[str, Any], raw: str) -> dict[str, Any] | None:
    """
    Recover a common small-model failure mode where JSON object fields are emitted
    as XML-like tags despite explicit JSON delimiters.
    """
    if not isinstance(raw, str) or not raw.strip():
        return None

    props = schema.get("properties") or {}
    if not isinstance(props, dict) or not props:
        return None

    payload = raw
    block = re.search(r"<\s*JSON_OBJECT\s*>(.*?)<\s*/\s*JSON_OBJECT\s*>", raw, re.I | re.S)
    if block:
        payload = block.group(1)

    obj: dict[str, Any] = {}
    for name in props:
        if not isinstance(name, str) or not name:
            continue
        pattern = rf"<\s*{re.escape(name)}\s*>(.*?)<\s*/\s*{re.escape(name)}\s*>"
        match = re.search(pattern, payload, re.I | re.S)
        if not match:
            continue
        value = html.unescape(match.group(1)).strip()
        obj[name] = None if value.lower() == "null" else value

    return obj or None


def validate_first_matching(schema: dict[str, Any], raw_output: str) -> dict[str, Any]:
    if raw_output is None:
        raise AppError(
            code="invalid_json",
            message="Model output was empty.",
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        )

    s = raw_output.strip()

    # Prefer delimited JSON if present
    if _JSON_BEGIN in s and _JSON_END in s:
        inner = ""
        try:
            inner = s.split(_JSON_BEGIN, 1)[1].split(_JSON_END, 1)[0].strip()
            inner = strip_wrapping_code_fences(inner)
            obj = json.loads(inner)
            if not isinstance(obj, dict):
                raise ValueError("Delimited JSON was not an object")
            validate_jsonschema(schema, obj)
            return obj
        except DependencyMissingError as e:
            raise AppError(code=e.code, message=e.message, status_code=500) from e
        except JSONSchemaValidationError as e:
            raise AppError(
                code=e.code,
                message=e.message,
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                extra={"errors": e.errors, "raw_preview": (raw_output or "")[:500]},
            ) from e
        except Exception:
            pass

        tag_obj = _schema_tag_object_candidate(schema, inner)
        if tag_obj is not None:
            candidates = [tag_obj] + iter_json_objects(s)
        else:
            candidates = iter_json_objects(s)
    else:
        tag_obj = _schema_tag_object_candidate(schema, s)
        if tag_obj is not None:
            candidates = [tag_obj] + iter_json_objects(s)
        else:
            candidates = iter_json_objects(s)

    if not candidates:
        raise AppError(
            code="invalid_json",
            message="Model output did not contain any JSON object.",
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            extra={"raw_preview": (raw_output or "")[:500]},
        )

    last_validation_error: Optional[JSONSchemaValidationError] = None

    for obj in candidates:
        try:
            validate_jsonschema(schema, obj)
            return obj
        except DependencyMissingError as e:
            raise AppError(code=e.code, message=e.message, status_code=500) from e
        except JSONSchemaValidationError as e:
            last_validation_error = e
            continue
        except Exception:
            continue

    extra: dict[str, Any] = {
        "raw_preview": (raw_output or "")[:500],
        "candidates_found": len(candidates),
    }
    if last_validation_error is not None:
        extra["errors"] = last_validation_error.errors

    raise AppError(
        code="schema_validation_failed",
        message="No JSON object in the model output conformed to the schema.",
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        extra=extra,
    )
