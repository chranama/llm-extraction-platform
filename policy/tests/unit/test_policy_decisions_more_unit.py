from __future__ import annotations

from pathlib import Path

from llm_policy.io.policy_decisions import (
    _decision_to_payload,
    _issue_to_dict,
    _status_value,
    default_policy_out_path,
    load_policy_decision,
)
from llm_policy.types.decision import Decision, DecisionStatus


class _IssueObj:
    code = "c1"
    message = "m1"
    context = "not-a-dict"


def test_status_value_normalization() -> None:
    assert _status_value(None) == "unknown"
    assert _status_value("allow") == "allow"
    assert _status_value(3) == "3"


def test_issue_to_dict_normalizes_context() -> None:
    out = _issue_to_dict(_IssueObj())
    assert out["code"] == "c1"
    assert out["message"] == "m1"
    assert out["context"] == {}


def test_decision_to_payload_generate_clamp_only_nulls(monkeypatch) -> None:
    monkeypatch.setattr("llm_policy.io.policy_decisions.validate_internal", lambda *_a, **_k: None)

    d = Decision(
        policy="p1",
        pipeline="generate_clamp_only",
        status=DecisionStatus.allow,
        enable_extract=True,
        generate_max_new_tokens_cap=128,
        thresholds_profile="extract/default",
        eval_run_dir="/tmp/run",
    )
    payload = _decision_to_payload(d)

    assert payload["pipeline"] == "generate_clamp_only"
    assert payload["enable_extract"] is None
    assert payload["thresholds_profile"] is None
    assert payload["eval_run_dir"] is None
    assert payload["generate_max_new_tokens_cap"] == 128


def test_decision_to_payload_fail_closed_extract_pipeline(monkeypatch) -> None:
    monkeypatch.setattr("llm_policy.io.policy_decisions.validate_internal", lambda *_a, **_k: None)

    d = Decision(
        policy="p1",
        pipeline="extract_only",
        status=DecisionStatus.deny,
        enable_extract=True,
        generate_max_new_tokens_cap=777,
        generate_thresholds_profile="generate/portable",
    )
    payload = _decision_to_payload(d)

    assert payload["ok"] is False
    assert payload["enable_extract"] is False
    assert payload["generate_max_new_tokens_cap"] is None
    assert payload["generate_thresholds_profile"] is None


def test_load_policy_decision_delegates(monkeypatch, tmp_path: Path) -> None:
    expected = object()
    p = tmp_path / "latest.json"
    monkeypatch.setattr(
        "llm_policy.io.policy_decisions.read_policy_decision",
        lambda path: expected if str(path) == str(p) else None,
    )

    assert load_policy_decision(p) is expected


def test_default_policy_out_path_without_env(monkeypatch) -> None:
    monkeypatch.delenv("POLICY_OUT_PATH", raising=False)
    p = default_policy_out_path()
    assert p.is_absolute()
    assert p.as_posix().endswith("policy_out/latest.json")
