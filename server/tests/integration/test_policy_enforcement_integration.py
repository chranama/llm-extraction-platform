# tests/integration/test_policy_enforcement_integration.py
from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write(p: Path, obj) -> None:
    p.write_text(json.dumps(obj), encoding="utf-8")


def _policy_v2(**overrides):
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
        "eval_run_dir": "/tmp/eval",
        "eval_task": "extract",
        "eval_run_id": "run-1",
        "model_id": None,
        "reasons": [],
        "warnings": [],
    }
    base.update(overrides)
    return base


def _clear_policy_snapshot(client) -> None:
    if hasattr(client, "app"):
        try:
            delattr(client.app.state, "policy_snapshot")
        except Exception:
            pass


@pytest.fixture(autouse=True)
def _force_lazy_model_mode(app):
    app.state.settings.model_load_mode = "lazy"
    app.state.model_load_mode = "lazy"
    yield


@pytest.mark.anyio
async def test_policy_disables_extract_blocks_endpoint(
    client, auth_headers, monkeypatch, tmp_path: Path
):
    p = tmp_path / "policy.json"
    _write(p, _policy_v2(enable_extract=False))
    monkeypatch.setenv("POLICY_DECISION_PATH", str(p))
    _clear_policy_snapshot(client)

    # Capability is enforced before schema load, so schema_id can be anything.
    payload = {"schema_id": "does_not_matter", "text": "hello"}
    r = await client.post("/v1/extract", json=payload, headers=auth_headers)
    assert r.status_code == 400

    body = r.json()
    assert body.get("code") == "capability_not_supported"
    # sanity check the merged caps include policy denial
    extra = body.get("extra") or {}
    caps = extra.get("model_capabilities") or {}
    assert caps.get("extract") is False


@pytest.mark.anyio
async def test_policy_disables_extract_reflected_in_models_endpoint(
    client, auth_headers, monkeypatch, tmp_path: Path
):
    p = tmp_path / "policy.json"
    _write(p, _policy_v2(enable_extract=False))
    monkeypatch.setenv("POLICY_DECISION_PATH", str(p))
    _clear_policy_snapshot(client)

    r = await client.get("/v1/models")
    assert r.status_code == 200
    data = r.json()
    assert "models" in data and len(data["models"]) >= 1

    # Default model should reflect extract=False
    default_id = data["default_model"]
    m = next(x for x in data["models"] if x["id"] == default_id)
    assert m["capabilities"]["extract"] is False


@pytest.mark.anyio
async def test_policy_invalid_file_fail_closed_blocks_extract(
    client, auth_headers, monkeypatch, tmp_path: Path
):
    p = tmp_path / "policy.json"
    p.write_text("{not-json", encoding="utf-8")
    monkeypatch.setenv("POLICY_DECISION_PATH", str(p))
    _clear_policy_snapshot(client)

    payload = {"schema_id": "whatever", "text": "hello"}
    r = await client.post("/v1/extract", json=payload, headers=auth_headers)
    assert r.status_code == 400
    assert r.json().get("code") == "capability_not_supported"
