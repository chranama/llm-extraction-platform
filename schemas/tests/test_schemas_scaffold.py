from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = REPO_ROOT / "schemas" / "internal"
FORMAT_CHECKER = FormatChecker()


def _load_schema(filename: str) -> dict:
    return json.loads((SCHEMA_DIR / filename).read_text(encoding="utf-8"))


def _assert_valid(schema_name: str, payload: dict) -> None:
    validator = Draft202012Validator(_load_schema(schema_name), format_checker=FORMAT_CHECKER)
    errors = sorted(validator.iter_errors(payload), key=lambda e: list(e.path))
    assert not errors, [e.message for e in errors]


def _assert_invalid(schema_name: str, payload: dict) -> None:
    validator = Draft202012Validator(_load_schema(schema_name), format_checker=FORMAT_CHECKER)
    errors = list(validator.iter_errors(payload))
    assert errors


@pytest.mark.parametrize(
    "schema_name",
    [
        "eval_run_summary_v2.schema.json",
        "eval_result_row_v2.schema.json",
        "policy_decision_v2.schema.json",
    ],
)
def test_v2_internal_schemas_are_valid_draft_2020_12(schema_name: str) -> None:
    Draft202012Validator.check_schema(_load_schema(schema_name))


def test_eval_run_summary_v2_accepts_valid_payload() -> None:
    payload = {
        "schema_version": "eval_run_summary_v2",
        "generated_at": "2026-02-28T00:00:00Z",
        "task": "extraction_sroie",
        "run_id": "run_1",
        "run_dir": "/tmp/eval/run_1",
        "passed": True,
        "metrics": {"schema_valid_rate": 0.99},
        "counts": {"examples_total": 10, "examples_ok": 9, "examples_failed": 1},
        "warnings": [],
    }
    _assert_valid("eval_run_summary_v2.schema.json", payload)


def test_eval_run_summary_v2_rejects_missing_metrics() -> None:
    payload = {
        "schema_version": "eval_run_summary_v2",
        "generated_at": "2026-02-28T00:00:00Z",
        "task": "extraction_sroie",
        "run_id": "run_1",
        "run_dir": "/tmp/eval/run_1",
        "passed": True,
    }
    _assert_invalid("eval_run_summary_v2.schema.json", payload)


def test_eval_result_row_v2_accepts_valid_payload() -> None:
    payload = {
        "schema_version": "eval_result_row_v2",
        "task": "extraction_sroie",
        "run_id": "run_1",
        "example_id": "example-001",
        "schema_id": "sroie_receipt_v1",
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "ok": False,
        "error": {"code": "timeout", "message": "upstream timeout"},
        "latency_ms": 123.4,
        "tokens": {"prompt": 100, "completion": 42},
    }
    _assert_valid("eval_result_row_v2.schema.json", payload)


def test_eval_result_row_v2_rejects_bad_error_object() -> None:
    payload = {
        "schema_version": "eval_result_row_v2",
        "task": "extraction_sroie",
        "run_id": "run_1",
        "ok": False,
        "error": {"message": "missing required code"},
    }
    _assert_invalid("eval_result_row_v2.schema.json", payload)


def test_policy_decision_v2_accepts_extract_only_payload() -> None:
    payload = {
        "schema_version": "policy_decision_v2",
        "generated_at": "2026-02-28T00:00:00Z",
        "policy": "extract_enablement",
        "pipeline": "extract_only",
        "status": "allow",
        "ok": True,
        "enable_extract": True,
        "generate_max_new_tokens_cap": None,
        "contract_errors": 0,
        "thresholds_profile": "extract/default",
        "thresholds_version": "v1",
        "generate_thresholds_profile": None,
        "eval_run_dir": "/tmp/eval/run_1",
        "eval_task": "extraction_sroie",
        "eval_run_id": "run_1",
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "reasons": [{"code": "ok", "message": "all gates passed", "context": {}}],
        "warnings": [],
        "metrics": {"schema_valid_rate": 0.99},
        "contract_warnings": 0,
    }
    _assert_valid("policy_decision_v2.schema.json", payload)


def test_policy_decision_v2_enforces_pipeline_status_rules() -> None:
    payload = {
        "schema_version": "policy_decision_v2",
        "generated_at": "2026-02-28T00:00:00Z",
        "policy": "extract_enablement",
        "pipeline": "extract_only",
        "status": "deny",
        "ok": True,
        "enable_extract": True,
        "generate_max_new_tokens_cap": None,
        "contract_errors": 0,
        "thresholds_profile": "extract/default",
        "thresholds_version": "v1",
        "generate_thresholds_profile": None,
        "eval_run_dir": "/tmp/eval/run_1",
        "eval_task": "extraction_sroie",
        "eval_run_id": "run_1",
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "reasons": [{"code": "failed", "message": "gate failed", "context": {}}],
        "warnings": [],
    }
    _assert_invalid("policy_decision_v2.schema.json", payload)
