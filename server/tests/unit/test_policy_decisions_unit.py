from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_server.io.policy_decisions import (
    PolicyDecisionSnapshot,
    _best_effort_parse_v2_json,
    _to_backend_snapshot,
    get_policy_snapshot,
    load_policy_decision_from_env,
    policy_capability_overrides,
    policy_generate_max_new_tokens_cap,
    reload_policy_snapshot,
)


@pytest.fixture(autouse=True)
def _set_schemas_root(monkeypatch: pytest.MonkeyPatch):
    repo_root = Path(__file__).resolve().parents[3]
    monkeypatch.setenv("SCHEMAS_ROOT", str(repo_root / "schemas"))


def _payload(**overrides):
    base = {
        "schema_version": "policy_decision_v2",
        "generated_at": "2026-01-01T00:00:00Z",
        "policy": "extract_enablement",
        "pipeline": "extract_only",
        "status": "allow",
        "ok": True,
        "enable_extract": True,
        "generate_max_new_tokens_cap": None,
        "contract_errors": 0,
        "thresholds_profile": "default",
        "thresholds_version": "v1",
        "generate_thresholds_profile": None,
        "eval_run_dir": "/tmp/eval-run",
        "eval_task": "extract",
        "eval_run_id": "run-1",
        "model_id": None,
        "reasons": [],
        "warnings": [],
    }
    base.update(overrides)
    return base


def _write(p: Path, obj) -> None:
    p.write_text(json.dumps(obj), encoding="utf-8")


def test_policy_no_env_path(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("POLICY_DECISION_PATH", raising=False)
    snap = load_policy_decision_from_env()
    assert snap.ok is True
    assert snap.model_id is None
    assert snap.enable_extract is None
    assert snap.source_path is None
    assert snap.error is None
    assert snap.raw == {}


def test_policy_missing_file_fail_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    p = tmp_path / "missing.json"
    monkeypatch.setenv("POLICY_DECISION_PATH", str(p))
    snap = load_policy_decision_from_env()
    assert snap.ok is False
    assert snap.enable_extract is False
    assert snap.error == "policy_decision_missing"
    assert snap.source_path == str(p)


def test_policy_invalid_json_fail_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    p = tmp_path / "bad.json"
    p.write_text("{not-json", encoding="utf-8")
    monkeypatch.setenv("POLICY_DECISION_PATH", str(p))
    snap = load_policy_decision_from_env()
    assert snap.ok is False
    assert snap.enable_extract is False
    assert snap.source_path == str(p)
    assert snap.error is not None
    assert snap.error.startswith(
        ("policy_decision_read_error:", "policy_decision_parse_error:")
    )


def test_policy_enable_extract_true(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    p = tmp_path / "ok.json"
    _write(p, _payload(enable_extract=True))
    monkeypatch.setenv("POLICY_DECISION_PATH", str(p))
    snap = load_policy_decision_from_env()
    assert snap.ok is True
    assert snap.enable_extract is True
    assert snap.error is None


def test_policy_contract_errors_nonzero_denies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    p = tmp_path / "deny.json"
    _write(p, _payload(ok=False, contract_errors=2, enable_extract=False))
    monkeypatch.setenv("POLICY_DECISION_PATH", str(p))
    snap = load_policy_decision_from_env()
    assert snap.ok is False
    assert snap.enable_extract is False
    assert snap.error in (None, "policy_decision_not_ok")


def test_policy_status_deny_denies(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    p = tmp_path / "deny.json"
    _write(p, _payload(status="deny", ok=False, enable_extract=False))
    monkeypatch.setenv("POLICY_DECISION_PATH", str(p))
    snap = load_policy_decision_from_env()
    assert snap.ok is False
    assert snap.enable_extract is False
    assert snap.error in (None, "policy_decision_not_ok")


class _Req:
    def __init__(self):
        class _State:
            ...

        class _App:
            ...

        self.app = _App()
        self.app.state = _State()


def test_policy_override_scoped_to_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    p = tmp_path / "scoped.json"
    _write(p, _payload(model_id="m1", enable_extract=False))
    monkeypatch.setenv("POLICY_DECISION_PATH", str(p))

    req = _Req()
    assert policy_capability_overrides("m1", request=req) == {"extract": False}
    assert policy_capability_overrides("m2", request=req) is None


def test_policy_invalid_file_fail_closed_for_all_models(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    p = tmp_path / "bad.json"
    p.write_text("{not-json", encoding="utf-8")
    monkeypatch.setenv("POLICY_DECISION_PATH", str(p))

    req = _Req()
    assert policy_capability_overrides("any", request=req) == {"extract": False}


def test_to_backend_snapshot_normalizes_cap_and_extract_semantics():
    s1 = PolicyDecisionSnapshot(
        ok=True,
        model_id="m1",
        enable_extract=True,
        generate_max_new_tokens_cap=32,
        raw={},
        source_path="/tmp/p.json",
        error=None,
    )
    # emulate contracts snapshot shape via SimpleNamespace
    from types import SimpleNamespace

    c1 = SimpleNamespace(
        ok=True,
        model_id="m1",
        enable_extract=True,
        generate_max_new_tokens_cap=32,
        raw={"x": 1},
        source_path="/tmp/p.json",
        error=None,
    )
    out1 = _to_backend_snapshot(c1)
    assert out1.ok is True
    assert out1.enable_extract is True
    assert out1.generate_max_new_tokens_cap == 32

    c2 = SimpleNamespace(
        ok=False,
        model_id=None,
        enable_extract=True,
        generate_max_new_tokens_cap=-1,
        raw={},
        source_path="/tmp/p.json",
        error="bad",
    )
    out2 = _to_backend_snapshot(c2)
    assert out2.ok is False
    assert out2.enable_extract is False
    assert out2.generate_max_new_tokens_cap is None


def test_best_effort_parse_v2_json(tmp_path: Path):
    p = tmp_path / "v2.json"
    _write(
        p,
        _payload(
            ok=True,
            enable_extract=False,
            generate_max_new_tokens_cap=64,
            model_id="m2",
        ),
    )
    out = _best_effort_parse_v2_json(p)
    assert out is not None
    assert out.ok is True
    assert out.model_id == "m2"
    assert out.enable_extract is False
    assert out.generate_max_new_tokens_cap == 64

    p2 = tmp_path / "notv2.json"
    _write(p2, {"schema_version": "other"})
    assert _best_effort_parse_v2_json(p2) is None


def test_policy_snapshot_cache_reload_and_generate_cap(monkeypatch: pytest.MonkeyPatch):
    req = _Req()
    snap1 = PolicyDecisionSnapshot(
        ok=True,
        model_id="m1",
        enable_extract=None,
        generate_max_new_tokens_cap=40,
        raw={},
        source_path="/tmp/p.json",
        error=None,
    )
    snap2 = PolicyDecisionSnapshot(
        ok=True,
        model_id="m2",
        enable_extract=None,
        generate_max_new_tokens_cap=20,
        raw={},
        source_path="/tmp/p2.json",
        error=None,
    )
    calls = {"n": 0}

    def _loader():
        calls["n"] += 1
        return snap1 if calls["n"] == 1 else snap2

    monkeypatch.setattr(
        "llm_server.io.policy_decisions.load_policy_decision_from_env",
        _loader,
    )

    g1 = get_policy_snapshot(req)
    g2 = get_policy_snapshot(req)
    assert g1 is g2
    assert calls["n"] == 1

    r = reload_policy_snapshot(req)
    assert r.model_id == "m2"
    assert calls["n"] == 2

    assert policy_generate_max_new_tokens_cap("m2", request=req) == 20
    assert policy_generate_max_new_tokens_cap("other", request=req) is None
