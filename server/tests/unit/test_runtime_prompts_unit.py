from __future__ import annotations

from llm_server.runtime.prompts import build_extraction_prompt, build_repair_prompt, schema_summary


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


def test_build_extraction_prompt_contains_contract_markers():
    prompt = build_extraction_prompt(
        "receipt_v1",
        {"properties": {"merchant": {"type": "string"}}},
        "Coffee shop receipt",
    )

    assert "SCHEMA_ID: receipt_v1" in prompt
    assert "<<<JSON>>>" in prompt
    assert "INPUT_TEXT:\nCoffee shop receipt" in prompt


def test_build_repair_prompt_contains_previous_output_and_error_hint():
    prompt = build_repair_prompt(
        "receipt_v1",
        {"properties": {"merchant": {"type": "string"}}},
        "Coffee shop receipt",
        '{"merchant": 123}',
        '{"code": "schema_invalid"}',
    )

    assert "PREVIOUS_OUTPUT:\n{\"merchant\": 123}" in prompt
    assert "ERROR_HINT:\n{\"code\": \"schema_invalid\"}" in prompt
