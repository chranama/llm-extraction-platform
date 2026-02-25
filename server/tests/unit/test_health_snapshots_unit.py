from __future__ import annotations

import types
from types import SimpleNamespace

import pytest

from llm_server.services.api_deps.health import snapshots as snap


def _req():
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))


def test_env_and_container_helpers(monkeypatch):
    monkeypatch.setenv("A", "  x ")
    assert snap._env_nonempty("A") == "x"
    monkeypatch.setenv("A", " ")
    assert snap._env_nonempty("A") is None

    monkeypatch.setenv("IN_DOCKER", "1")
    assert snap._in_container_best_effort() is True


def test_backend_model_info_best_effort():
    ok_obj = SimpleNamespace(model_info=lambda: {"ok": True})
    assert snap._backend_model_info_best_effort(ok_obj) == {"ok": True}

    raw_obj = SimpleNamespace(model_info=lambda: "raw")
    assert snap._backend_model_info_best_effort(raw_obj) == {"raw": "raw"}

    bad_obj = SimpleNamespace(model_info=lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    out = snap._backend_model_info_best_effort(bad_obj)
    assert out["ok"] is False


def test_torch_accel_snapshot_paths(monkeypatch):
    class _Cuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 2

    class _Mps:
        @staticmethod
        def is_available():
            return True

    fake_torch = types.SimpleNamespace(cuda=_Cuda(), backends=types.SimpleNamespace(mps=_Mps()))
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
    out = snap._torch_accel_snapshot()
    assert out["torch_present"] is True
    assert out["cuda_available"] is True
    assert out["cuda_device_count"] == 2
    assert out["mps_available"] is True


def test_policy_summary_and_generate_gate_snapshot(monkeypatch):
    monkeypatch.setattr(snap, "get_policy_snapshot", lambda request: {"x": 1}, raising=True)
    monkeypatch.setattr(snap, "policy_snapshot_summary", lambda s: {"ok": bool(s)}, raising=True)
    assert snap.policy_summary(_req()) == {"ok": True}

    gate = SimpleNamespace(
        snapshot=lambda: SimpleNamespace(
            enabled=True,
            max_concurrent=3,
            max_queue=10,
            timeout_seconds=5.0,
            in_flight_estimate=1,
            queue_depth_estimate=2,
        )
    )
    monkeypatch.setattr(snap, "get_generate_gate", lambda: gate, raising=True)
    gs = snap.generate_gate_snapshot()
    assert gs["enabled"] is True
    assert gs["max_queue"] == 10

    monkeypatch.setattr(snap, "get_generate_gate", lambda: (_ for _ in ()).throw(RuntimeError("x")), raising=True)
    assert snap.generate_gate_snapshot() == {"error": "unavailable"}


def test_deployment_key_from_app_state_and_deployment_metadata(monkeypatch):
    req = _req()
    req.app.state.llm = SimpleNamespace(_meta={"m1": {"deployment_key": "dk-meta"}})
    req.app.state.models_config = SimpleNamespace(
        defaults={"selected_profile": "host-transformers"},
        models=[SimpleNamespace(id="m1", deployment_key="dk-cfg")],
    )
    monkeypatch.delenv("DEPLOYMENT_KEY", raising=False)
    assert snap._deployment_key_from_app_state(req, model_id="m1") == "dk-meta"

    monkeypatch.setattr(snap, "resolve_default_model_id_and_backend_obj", lambda request: ("m1", "transformers", SimpleNamespace(model_info=lambda: {"ok": True})), raising=True)
    monkeypatch.setattr(snap, "_in_container_best_effort", lambda: False, raising=True)
    monkeypatch.setattr(snap, "_torch_accel_snapshot", lambda: {"torch_present": False}, raising=True)
    monkeypatch.setenv("APP_PROFILE", "host")
    dep = snap.deployment_metadata_snapshot(req)
    assert dep["ok"] is True
    assert dep["deployment_key"] == "dk-meta"
    assert dep["routing"]["default_model_id"] == "m1"


def test_assessed_gate_snapshot_paths(monkeypatch):
    req = _req()
    sp = SimpleNamespace(
        id="m1",
        deployment_key="dk1",
        assessment={"require_assessed_gate": True},
        capabilities_effective={"extract": {"assessed": False, "status": "blocked", "reason": "no eval", "details": {"a": 1}}},
    )
    cfg = SimpleNamespace(models=[sp])

    monkeypatch.setattr(snap, "resolve_default_model_id_and_backend_obj", lambda request: ("m1", "transformers", None), raising=True)
    monkeypatch.setattr(snap, "models_config_from_request", lambda request: cfg, raising=True)
    out = snap.assessed_gate_snapshot(req)
    assert out["ok"] is True
    assert out["snapshot"]["required"] is True
    assert out["snapshot"]["status"] == "blocked"

    monkeypatch.setattr(snap, "models_config_from_request", lambda request: SimpleNamespace(models=[]), raising=True)
    out2 = snap.assessed_gate_snapshot(req)
    assert out2["snapshot"]["error"] == "model_spec_not_found"

    monkeypatch.setattr(snap, "models_config_from_request", lambda request: (_ for _ in ()).throw(RuntimeError("bad")), raising=True)
    out3 = snap.assessed_gate_snapshot(req)
    assert out3["ok"] is False
