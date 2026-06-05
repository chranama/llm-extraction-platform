from __future__ import annotations

from llm_server.runtime.prompts import (
    build_extraction_prompt,
    build_repair_prompt,
    schema_example_object,
    schema_summary,
)


def test_schema_summary_includes_required_fields_and_constraints():
    summary = schema_summary(
        {
            "required": ["merchant"],
            "properties": {
                "merchant": {"type": "string", "description": "Store name"},
                "total": {"type": "number", "pattern": "^\\d+\\.\\d{2}$"},
            },
            "additionalProperties": False,
        }
    )

    assert "REQUIRED_FIELDS: merchant" in summary
    assert "- merchant: string" in summary
    assert "pattern=^\\d+\\.\\d{2}$" in summary
    assert "additionalProperties=false" in summary


def test_schema_example_object_uses_json_object_shape():
    example = schema_example_object(
        {
            "properties": {
                "merchant": {"type": ["string", "null"]},
                "total": {"type": "number"},
            }
        }
    )

    assert example == '{"merchant": "Example Merchant", "total": 0}'


def test_build_extraction_prompt_contains_contract_markers():
    prompt = build_extraction_prompt(
        "receipt_v1",
        {"properties": {"merchant": {"type": "string"}}},
        "Coffee shop receipt",
    )

    assert "SCHEMA_ID: receipt_v1" in prompt
    assert "<<<JSON>>>" in prompt
    assert '{"merchant": "Example Merchant"}' in prompt
    assert "format example only; do not copy its example values" in prompt
    assert "Do not use XML tags or tag-style field wrappers." in prompt
    assert "<JSON_OBJECT>" not in prompt
    assert "INPUT_TEXT:\nCoffee shop receipt" in prompt


def test_build_repair_prompt_contains_previous_output_and_error_hint():
    prompt = build_repair_prompt(
        "receipt_v1",
        {"properties": {"merchant": {"type": "string"}}},
        "Coffee shop receipt",
        '{"merchant": 123}',
        '{"code": "schema_invalid"}',
    )

    assert 'PREVIOUS_OUTPUT:\n{"merchant": 123}' in prompt
    assert 'ERROR_HINT:\n{"code": "schema_invalid"}' in prompt
    assert "Do not echo PREVIOUS_OUTPUT or ERROR_HINT." in prompt
    assert "Do not use XML tags or tag-style field wrappers." in prompt
    assert "<JSON_OBJECT>" not in prompt
