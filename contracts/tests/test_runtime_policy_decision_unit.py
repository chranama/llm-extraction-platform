from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_contracts.runtime.policy_decision import (
    parse_policy_decision,
    read_policy_decision,
    write_policy_decision,
)


def _policy_payload_extract_only(**overrides: object) -> dict:
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
        "contract_warnings": 0,
        "thresholds_profile": "extract/default",
        "thresholds_version": "v1",
        "generate_thresholds_profile": None,
        "eval_run_dir": "/tmp/eval/run_1",
        "eval_task": "extraction_sroie",
        "eval_run_id": "run_1",
        "model_id": "m1",
        "reasons": [{"code": "ok", "message": "ok", "context": {}}],
        "warnings": [],
        "metrics": {},
    }
    payload.update(overrides)
    return payload


def test_parse_policy_decision_extract_allow_roundtrip() -> None:
    snap = parse_policy_decision(_policy_payload_extract_only())
    assert snap.ok is True
    assert snap.pipeline == "extract_only"
    assert snap.enable_extract is True


def test_parse_policy_decision_generate_only_keeps_enable_extract_none() -> None:
    payload = _policy_payload_extract_only(
        pipeline="generate_clamp_only",
        status="allow",
        ok=True,
        enable_extract=None,
        thresholds_profile=None,
        thresholds_version=None,
        eval_run_dir=None,
        eval_task=None,
        eval_run_id=None,
        generate_thresholds_profile="generate/portable",
    )
    snap = parse_policy_decision(payload)
    assert snap.pipeline == "generate_clamp_only"
    assert snap.enable_extract is None
    assert snap.ok is True


def test_parse_policy_decision_extract_fail_closed_when_contract_errors() -> None:
    payload = _policy_payload_extract_only(
        status="deny",
        ok=False,
        enable_extract=False,
        contract_errors=2,
    )
    snap = parse_policy_decision(payload)
    assert snap.ok is False
    assert snap.enable_extract is False
    assert snap.contract_errors == 2


def test_write_policy_decision_rejects_wrong_schema_version(tmp_path: Path) -> None:
    bad = _policy_payload_extract_only(schema_version="policy_decision_v1")
    with pytest.raises(ValueError, match="Unsupported policy decision schema_version"):
        write_policy_decision(tmp_path / "policy.json", bad)


def test_read_policy_decision_fail_closed_on_invalid_payload(tmp_path: Path) -> None:
    path = tmp_path / "policy.json"
    path.write_text(json.dumps({"schema_version": "policy_decision_v2"}), encoding="utf-8")

    snap = read_policy_decision(path)
    assert snap.ok is False
    assert snap.pipeline == "unknown"
    assert snap.contract_errors == 1
    assert snap.error and "policy_decision_parse_error" in snap.error


def test_write_then_read_policy_decision_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "policy.json"
    payload = _policy_payload_extract_only()
    write_policy_decision(path, payload)

    snap = read_policy_decision(path)
    assert snap.ok is True
    assert snap.schema_version == "policy_decision_v2"
    assert snap.eval_run_id == "run_1"
