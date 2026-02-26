from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_policy.io.policy_decisions import (
    default_policy_out_path,
    render_policy_decision_json,
    write_policy_decision_artifact,
)
from llm_policy.types.decision import Decision, DecisionStatus


@pytest.fixture(autouse=True)
def _disable_internal_schema_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("llm_policy.io.policy_decisions.validate_internal", lambda *_a, **_k: None)

    def _fake_write_policy_decision(path, payload):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return p

    monkeypatch.setattr(
        "llm_policy.io.policy_decisions.write_policy_decision", _fake_write_policy_decision
    )


def test_render_policy_decision_json_is_v2_contract_extract() -> None:
    d = Decision(
        policy="extract_enablement",
        pipeline="extract_only",
        status=DecisionStatus.allow,
        enable_extract=True,
    )
    s = render_policy_decision_json(d)
    obj = json.loads(s)

    assert obj["schema_version"] == "policy_decision_v2"
    assert obj["ok"] is True
    assert obj["enable_extract"] is True
    assert obj["pipeline"] == "extract_only"
    assert "generated_at" in obj


def test_write_policy_decision_artifact_writes_atomically(tmp_path: Path) -> None:
    out = tmp_path / "policy_out" / "latest.json"
    d = Decision(
        policy="extract_enablement",
        pipeline="extract_only",
        status=DecisionStatus.allow,
        enable_extract=True,
    )

    p = write_policy_decision_artifact(d, out)
    assert p == out
    assert out.exists()

    obj = json.loads(out.read_text(encoding="utf-8"))
    assert obj["schema_version"] == "policy_decision_v2"


def test_default_policy_out_path_env_override(monkeypatch):
    monkeypatch.setenv("POLICY_OUT_PATH", "/tmp/policy/latest.json")
    p = default_policy_out_path()
    assert p.as_posix().endswith("/tmp/policy/latest.json")
