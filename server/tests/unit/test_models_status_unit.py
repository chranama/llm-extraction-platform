from __future__ import annotations

from types import SimpleNamespace

from llm_server.services.api_deps.models import status as st


def _req():
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))


def test_compute_public_models_status_off_mode(monkeypatch):
    monkeypatch.setattr(st, "effective_model_load_mode_from_request", lambda request: "off", raising=True)
    monkeypatch.setattr(st, "model_flags_from_app_state", lambda request: (False, "err", None, None), raising=True)
    monkeypatch.setattr(st, "resolve_default_model_id_and_backend_obj", lambda request: ("m1", "transformers", object()), raising=True)

    out = st.compute_public_models_status(_req())
    assert out["status"] == "ok"
    assert out["model_load_mode"] == "off"
    assert out["default_model_id"] == "m1"


def test_compute_public_models_status_non_off_degraded_when_error(monkeypatch):
    monkeypatch.setattr(st, "effective_model_load_mode_from_request", lambda request: "lazy", raising=True)
    monkeypatch.setattr(st, "model_flags_from_app_state", lambda request: (True, "load fail", "m1", "m1"), raising=True)
    monkeypatch.setattr(st, "resolve_default_model_id_and_backend_obj", lambda request: ("m1", "remote", object()), raising=True)

    out = st.compute_public_models_status(_req())
    assert out["status"] == "degraded"
    assert out["model_error"] == "load fail"
    assert out["runtime_default_model_id"] == "m1"
