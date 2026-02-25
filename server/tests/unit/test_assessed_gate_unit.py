from __future__ import annotations

from types import SimpleNamespace

import pytest

from llm_server.core.errors import AppError
from llm_server.services.api_deps.enforcement import assessed_gate as gate


def _req():
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))


def test_deployment_key_from_snapshot_and_request(monkeypatch):
    dep = {
        "profiles": {"app": "host"},
        "container": True,
        "platform": {"system": "Darwin", "machine": "arm64"},
        "accelerators": {"torch_present": True, "cuda_available": False, "cuda_device_count": 0, "mps_available": True},
    }
    k1 = gate.deployment_key_from_snapshot(dep)
    k2 = gate.deployment_key_from_snapshot(dep)
    assert isinstance(k1, str) and len(k1) == 16
    assert k1 == k2

    monkeypatch.setattr(gate, "deployment_metadata_snapshot", lambda request: {"deployment_key": "  dk-1  "}, raising=True)
    assert gate.deployment_key_from_request(_req()) == "dk-1"

    monkeypatch.setattr(gate, "deployment_metadata_snapshot", lambda request: dep, raising=True)
    assert len(gate.deployment_key_from_request(_req())) == 16

    def _boom(_request):
        raise RuntimeError("oops")

    monkeypatch.setattr(gate, "deployment_metadata_snapshot", _boom, raising=True)
    assert gate.deployment_key_from_request(_req()) == "unknown"


def test_cap_assessed_from_capability_raw_parsing():
    a, s, extra = gate._cap_assessed_from_capability_raw(True)
    assert (a, s, extra) == (None, None, {})

    a2, s2, e2 = gate._cap_assessed_from_capability_raw(
        {
            "assessed": True,
            "status": "ALLOWED",
            "reason": "ok",
            "assessed_at_utc": "2026-01-01T00:00:00Z",
            "details": {"x": 1},
        }
    )
    assert a2 is True
    assert s2 == "allowed"
    assert e2["reason"] == "ok"
    assert e2["details"] == {"x": 1}

    a3, s3, _ = gate._cap_assessed_from_capability_raw({"status": "invalid"})
    assert a3 is None
    assert s3 is None


def test_assessed_snapshot_from_models_yaml_defaults_when_missing_spec(monkeypatch):
    monkeypatch.setattr(gate, "_get_model_spec", lambda request, model_id: None, raising=True)
    snap = gate._assessed_snapshot_from_models_yaml(request=_req(), model_id="m1", capability="extract")
    assert snap["required"] is False
    assert snap["status"] == "allowed"
    assert snap["selected_deployment_key"] is None
    assert snap["source"] == "models.yaml"


def test_assessed_snapshot_model_and_capability_overrides(monkeypatch):
    sp = SimpleNamespace(
        id="m2",
        deployment_key="dk-model",
        assessment={
            "require_assessed_gate": True,
            "status": "unknown",
            "reason": "model-level",
        },
        capabilities_effective={
            "extract": {
                "assessed": False,
                "status": "blocked",
                "reason": "cap-level",
                "details": {"risk": "high"},
            }
        },
    )
    monkeypatch.setattr(gate, "_get_model_spec", lambda request, model_id: sp, raising=True)
    snap = gate._assessed_snapshot_from_models_yaml(request=_req(), model_id="m2", capability="extract")
    assert snap["required"] is True
    assert snap["status"] == "blocked"
    assert snap["assessed"] is False
    assert snap["reason"] == "cap-level"
    assert snap["details"]["risk"] == "high"
    assert snap["selected_deployment_key"] == "dk-model"


def test_require_assessed_gate_allows_and_blocks(monkeypatch):
    req = _req()

    monkeypatch.setattr(
        gate,
        "_assessed_snapshot_from_models_yaml",
        lambda **kwargs: {"required": False, "status": "unknown"},
        raising=True,
    )
    gate.require_assessed_gate(request=req, model_id="m1", capability="extract")

    monkeypatch.setattr(
        gate,
        "_assessed_snapshot_from_models_yaml",
        lambda **kwargs: {
            "required": True,
            "status": "allowed",
            "selected_deployment_key": "dk-expected",
            "reason": None,
        },
        raising=True,
    )
    gate.require_assessed_gate(request=req, model_id="m1", capability="extract")

    monkeypatch.setattr(
        gate,
        "_assessed_snapshot_from_models_yaml",
        lambda **kwargs: {
            "required": True,
            "status": "blocked",
            "selected_deployment_key": "dk-expected",
            "reason": "policy denied",
        },
        raising=True,
    )
    monkeypatch.setattr(gate, "deployment_key_from_request", lambda request: "dk-current", raising=True)

    with pytest.raises(AppError) as e:
        gate.require_assessed_gate(request=req, model_id="m1", capability="extract")
    assert e.value.code == "assessed_gate_blocked"
    assert e.value.status_code == 503
    assert e.value.extra["deployment"]["deployment_key_matches"] is False
