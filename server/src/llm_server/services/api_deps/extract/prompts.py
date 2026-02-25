# server/src/llm_server/services/api_deps/extract/prompts.py
from __future__ import annotations

from typing import Any, Dict

from llm_server.services.api_deps.extract.constants import _JSON_BEGIN, _JSON_END


def schema_summary(schema: dict[str, Any]) -> str:
    required = schema.get("required") or []
    props = schema.get("properties") or {}

    lines: list[str] = []
    if required:
        lines.append(f"REQUIRED_FIELDS: {', '.join(required)}")

    lines.append("FIELDS:")
    for k, v in props.items():
        if not isinstance(v, dict):
            continue
        t = v.get("type", "any")
        enum = v.get("enum")
        pat = v.get("pattern")
        desc = v.get("description")

        pieces = [f"- {k}: {t}"]
        if enum:
            pieces.append(f"enum={enum}")
        if pat:
            pieces.append(f"pattern={pat}")
        if desc:
            pieces.append(f"desc={str(desc)[:80]}")
        lines.append("  " + " | ".join(pieces))

    ap = schema.get("additionalProperties", None)
    if ap is False:
        lines.append("CONSTRAINT: additionalProperties=false (no extra keys).")

    return "\n".join(lines)


def build_extraction_prompt(schema_id: str, schema: dict[str, Any], text: str) -> str:
    summary = schema_summary(schema)
    return (
        "You are a structured information extraction engine.\n"
        "Return ONLY a JSON object that matches the contract below.\n"
        "No markdown. No code fences. No commentary.\n"
        "If a value is unknown: omit the field unless it is REQUIRED.\n"
        "If a REQUIRED field is missing in the text: set it to null.\n\n"
        f"OUTPUT FORMAT:\n{_JSON_BEGIN}\n<JSON_OBJECT>\n{_JSON_END}\n\n"
        f"SCHEMA_ID: {schema_id}\n"
        f"{summary}\n\n"
        f"INPUT_TEXT:\n{text}\n"
    )


def build_repair_prompt(
    schema_id: str,
    schema: dict[str, Any],
    text: str,
    bad_output: str,
    error_hint: str,
) -> str:
    summary = schema_summary(schema)
    return (
        "Your previous output did NOT match the contract.\n"
        "Fix it. Return ONLY the corrected JSON object.\n"
        "No markdown. No code fences. No commentary.\n\n"
        f"OUTPUT FORMAT:\n{_JSON_BEGIN}\n<JSON_OBJECT>\n{_JSON_END}\n\n"
        f"SCHEMA_ID: {schema_id}\n"
        f"{summary}\n\n"
        f"INPUT_TEXT:\n{text}\n\n"
        f"PREVIOUS_OUTPUT:\n{bad_output}\n\n"
        f"ERROR_HINT:\n{error_hint}\n"
    )