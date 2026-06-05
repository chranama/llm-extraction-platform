from __future__ import annotations

import json
from typing import Any

_JSON_BEGIN = "<<<JSON>>>"
_JSON_END = "<<<END>>>"


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


def _example_value(name: str, spec: dict[str, Any]) -> Any:
    enum = spec.get("enum")
    if isinstance(enum, list) and enum:
        return enum[0]

    t = spec.get("type")
    types = t if isinstance(t, list) else [t]
    non_null_types = [x for x in types if x != "null"]
    if not non_null_types and "null" in types:
        return None

    if "string" in non_null_types:
        lname = name.lower()
        if "date" in lname:
            return "2024-01-01"
        if "total" in lname or "amount" in lname or "price" in lname:
            return "$0.00"
        if "address" in lname:
            return "123 Example St"
        if "company" in lname or "merchant" in lname:
            return "Example Merchant"
        return "example"
    if "integer" in non_null_types:
        return 0
    if "number" in non_null_types:
        return 0
    if "boolean" in non_null_types:
        return False
    if "array" in non_null_types:
        return []
    if "object" in non_null_types:
        return {}
    return None


def schema_example_object(schema: dict[str, Any]) -> str:
    props = schema.get("properties") or {}
    example: dict[str, Any] = {}
    if isinstance(props, dict):
        for k, v in props.items():
            if isinstance(k, str) and isinstance(v, dict):
                example[k] = _example_value(k, v)
    return json.dumps(example, ensure_ascii=False, separators=(", ", ": "))


def build_extraction_prompt(schema_id: str, schema: dict[str, Any], text: str) -> str:
    summary = schema_summary(schema)
    example = schema_example_object(schema)
    return (
        "You are a structured information extraction engine.\n"
        "Return ONLY a JSON object that matches the contract below.\n"
        "No markdown. No code fences. No commentary.\n"
        "Do not use XML tags or tag-style field wrappers.\n"
        "If a value is unknown: omit the field unless it is REQUIRED.\n"
        "If a REQUIRED field is missing in the text: set it to null.\n\n"
        "OUTPUT FORMAT:\n"
        "The object below is a format example only; do not copy its example values.\n"
        f"Return exactly one JSON object between these markers:\n{_JSON_BEGIN}\n"
        f"{example}\n{_JSON_END}\n\n"
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
    example = schema_example_object(schema)
    return (
        "Your previous output did NOT match the contract.\n"
        "Fix it. Return ONLY the corrected JSON object.\n"
        "No markdown. No code fences. No commentary.\n\n"
        "Do not echo PREVIOUS_OUTPUT or ERROR_HINT.\n"
        "Do not use XML tags or tag-style field wrappers.\n\n"
        "OUTPUT FORMAT:\n"
        "The object below is a format example only; do not copy its example values.\n"
        f"Return exactly one JSON object between these markers:\n{_JSON_BEGIN}\n"
        f"{example}\n{_JSON_END}\n\n"
        f"SCHEMA_ID: {schema_id}\n"
        f"{summary}\n\n"
        f"INPUT_TEXT:\n{text}\n\n"
        f"PREVIOUS_OUTPUT:\n{bad_output}\n\n"
        f"ERROR_HINT:\n{error_hint}\n"
    )
