from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from llm_server.api import health as health_api


def _request():
    state = SimpleNamespace(
        llm=None,
        model_loaded=False,
        model_error=None,
        loaded_model_id=None,
        runtime_default_model_id=None,
        runtime_model_loader=None,
        models_config=None,
    )
    return SimpleNamespace(app=SimpleNamespace(state=state))


def _payload(resp):
    return json.loads(resp.body.decode("utf-8"))


def test_llm_state_variants():
    assert health_api._llm_state(None) == "not initialized"

    loaded_obj = SimpleNamespace(is_loaded=lambda: True)
    assert health_api._llm_state(loaded_obj) == "loaded"

    err_obj = SimpleNamespace(is_loaded=lambda: (_ for _ in ()).throw(RuntimeError("x")))
    assert health_api._llm_state(err_obj) == "unknown"

    attr_obj = SimpleNamespace(loaded=False)
    assert health_api._llm_state(attr_obj) == "not loaded"

    model_obj = SimpleNamespace(model=object())
    assert health_api._llm_state(model_obj) == "loaded"


def test_model_required_for_readyz(monkeypatch):
    monkeypatch.setattr(health_api, "settings_from_request", lambda request: SimpleNamespace(require_model_ready=True), raising=True)
    assert health_api._model_required_for_readyz(_request()) is True


@pytest.mark.anyio
async def test_readyz_when_model_not_required(monkeypatch):
    req = _request()
    req.app.state.llm = SimpleNamespace(is_loaded=lambda: True)
    req.app.state.runtime_model_loader = object()
    req.app.state.models_config = object()

    monkeypatch.setattr(health_api, "settings_from_request", lambda request: SimpleNamespace(db_instance="dbA", require_model_ready=False), raising=True)
    async def _db_ok(_session):
        return True, "ok"

    async def _redis_ok(_request):
        return True, "ok"

    monkeypatch.setattr(health_api, "db_check", _db_ok, raising=True)
    monkeypatch.setattr(health_api, "redis_check", _redis_ok, raising=True)
    monkeypatch.setattr(health_api, "effective_model_load_mode_from_request", lambda request: "lazy", raising=True)
    monkeypatch.setattr(health_api, "model_flags_from_app_state", lambda request: (True, None, "m1", "m1"), raising=True)
    monkeypatch.setattr(health_api, "resolve_default_model_id_and_backend_obj", lambda request: ("m1", "transformers", object()), raising=True)
    monkeypatch.setattr(health_api, "per_model_readiness_mode", lambda request, model_id: "generate", raising=True)
    monkeypatch.setattr(health_api, "policy_summary", lambda request: {"ok": True}, raising=True)
    monkeypatch.setattr(health_api, "generate_gate_snapshot", lambda: {"enabled": True}, raising=True)
    monkeypatch.setattr(health_api, "assessed_gate_snapshot", lambda request: {"status": "allowed"}, raising=True)
    monkeypatch.setattr(health_api, "deployment_metadata_snapshot", lambda request: {"deployment_key": "dk"}, raising=True)

    async def _model_ready(_request):
        return False, "not ready", {"x": 1}

    monkeypatch.setattr(health_api, "model_ready_check_async", _model_ready, raising=True)

    resp = await health_api.readyz(req, session=object())
    data = _payload(resp)
    assert resp.status_code == 200
    assert data["status"] == "ready"
    assert data["model"]["required"] is False
    assert data["model"]["status"] == "skipped"
    assert data["runtime"]["runtime_model_loader_present"] is True


@pytest.mark.anyio
async def test_readyz_model_required_failure(monkeypatch):
    req = _request()
    req.app.state.llm = object()
    monkeypatch.setattr(health_api, "settings_from_request", lambda request: SimpleNamespace(db_instance="dbB", require_model_ready=True), raising=True)
    async def _db_ok(_session):
        return True, "ok"

    async def _redis_ok(_request):
        return True, "ok"

    monkeypatch.setattr(health_api, "db_check", _db_ok, raising=True)
    monkeypatch.setattr(health_api, "redis_check", _redis_ok, raising=True)
    monkeypatch.setattr(health_api, "effective_model_load_mode_from_request", lambda request: "lazy", raising=True)
    monkeypatch.setattr(health_api, "model_flags_from_app_state", lambda request: (False, "err", None, None), raising=True)
    monkeypatch.setattr(health_api, "resolve_default_model_id_and_backend_obj", lambda request: ("m2", "llamacpp", object()), raising=True)
    monkeypatch.setattr(health_api, "per_model_readiness_mode", lambda request, model_id: "probe", raising=True)
    monkeypatch.setattr(health_api, "policy_summary", lambda request: {}, raising=True)
    monkeypatch.setattr(health_api, "generate_gate_snapshot", lambda: {}, raising=True)
    monkeypatch.setattr(health_api, "assessed_gate_snapshot", lambda request: {}, raising=True)
    monkeypatch.setattr(health_api, "deployment_metadata_snapshot", lambda request: {}, raising=True)

    async def _model_ready(_request):
        return False, "dependency down", {"why": "offline"}

    monkeypatch.setattr(health_api, "model_ready_check_async", _model_ready, raising=True)

    resp = await health_api.readyz(req, session=object())
    data = _payload(resp)
    assert resp.status_code == 503
    assert data["status"] == "not ready"
    assert data["model"]["required"] is True
    assert data["model"]["ok"] is False
    assert data["model"]["details"]["why"] == "offline"


@pytest.mark.anyio
async def test_modelz_success_and_failure(monkeypatch):
    req = _request()
    req.app.state.llm = object()
    monkeypatch.setattr(health_api, "settings_from_request", lambda request: SimpleNamespace(db_instance="dbC"), raising=True)
    monkeypatch.setattr(health_api, "effective_model_load_mode_from_request", lambda request: "off", raising=True)
    monkeypatch.setattr(health_api, "model_flags_from_app_state", lambda request: (False, None, None, "m3"), raising=True)
    monkeypatch.setattr(health_api, "resolve_default_model_id_and_backend_obj", lambda request: ("m3", "remote", object()), raising=True)
    monkeypatch.setattr(health_api, "per_model_readiness_mode", lambda request, model_id: "generate", raising=True)
    monkeypatch.setattr(health_api, "policy_summary", lambda request: {}, raising=True)
    monkeypatch.setattr(health_api, "generate_gate_snapshot", lambda: {}, raising=True)
    monkeypatch.setattr(health_api, "assessed_gate_snapshot", lambda request: {}, raising=True)
    monkeypatch.setattr(health_api, "deployment_metadata_snapshot", lambda request: {}, raising=True)

    async def _ok(_request):
        return True, "ok", {"k": 1}

    monkeypatch.setattr(health_api, "model_ready_check_async", _ok, raising=True)
    resp_ok = await health_api.modelz(req)
    data_ok = _payload(resp_ok)
    assert resp_ok.status_code == 200
    assert data_ok["status"] == "ready"
    assert data_ok["llm"] == "disabled"

    async def _bad(_request):
        return False, "not ready", {"k": 2}

    monkeypatch.setattr(health_api, "model_ready_check_async", _bad, raising=True)
    resp_bad = await health_api.modelz(req)
    data_bad = _payload(resp_bad)
    assert resp_bad.status_code == 503
    assert data_bad["reason"] == "not ready"
    assert data_bad["dependency"]["backend"]["details"]["k"] == 2
