from __future__ import annotations

from types import SimpleNamespace

import pytest

from llm_server.core.errors import AppError
from llm_server.services.api_deps.models import listing as ml
from llm_server.services.llm_runtime.llm_registry import MultiModelManager


def _req():
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(model_loaded=False)))


class _Backend:
    def __init__(self, model_id: str, backend_name: str = "transformers"):
        self.model_id = model_id
        self.backend_name = backend_name


def test_list_models_payload_off_mode(monkeypatch):
    req = _req()
    monkeypatch.setattr(ml, "settings_from_request", lambda request: SimpleNamespace(model_id="m-default"), raising=True)
    monkeypatch.setattr(ml, "deployment_capabilities", lambda request: {"generate": True}, raising=True)
    monkeypatch.setattr(ml, "effective_model_load_mode_from_request", lambda request: "off", raising=True)
    monkeypatch.setattr(ml, "allowed_model_ids", lambda request=None: ["m-a"], raising=True)
    monkeypatch.setattr(ml, "effective_capabilities", lambda mid, request=None: {"extract": mid == "m-default"}, raising=True)

    out = ml.list_models_payload(request=req, llm=None)
    assert out["default_model"] == "m-default"
    assert [x["id"] for x in out["models"]] == ["m-default", "m-a"]
    assert all(x["load_mode"] == "off" for x in out["models"])


def test_list_models_payload_non_off_llm_missing_raises(monkeypatch):
    req = _req()
    monkeypatch.setattr(ml, "settings_from_request", lambda request: SimpleNamespace(model_id="m1"), raising=True)
    monkeypatch.setattr(ml, "deployment_capabilities", lambda request: {}, raising=True)
    monkeypatch.setattr(ml, "effective_model_load_mode_from_request", lambda request: "lazy", raising=True)

    with pytest.raises(AppError) as e:
        ml.list_models_payload(request=req, llm=None)
    assert e.value.code == "llm_unavailable"


def test_list_models_payload_multimodel_status_and_fallback(monkeypatch):
    req = _req()
    mm = MultiModelManager(
        models={"m1": _Backend("m1", "remote"), "m2": _Backend("m2", "llamacpp")},
        default_id="m2",
    )
    monkeypatch.setattr(ml, "settings_from_request", lambda request: SimpleNamespace(model_id="m1"), raising=True)
    monkeypatch.setattr(ml, "deployment_capabilities", lambda request: {"extract": True}, raising=True)
    monkeypatch.setattr(ml, "effective_model_load_mode_from_request", lambda request: "lazy", raising=True)
    monkeypatch.setattr(ml, "effective_capabilities", lambda mid, request=None: {"generate": True}, raising=True)

    out = ml.list_models_payload(request=req, llm=mm)
    assert out["default_model"] == "m2"
    by_id = {x["id"]: x for x in out["models"]}
    assert by_id["m2"]["default"] is True
    assert by_id["m1"]["backend"] == "_Backend"

    # status() failure path still falls back to backend introspection
    monkeypatch.setattr(mm, "status", lambda: (_ for _ in ()).throw(RuntimeError("boom")), raising=True)
    out2 = ml.list_models_payload(request=req, llm=mm)
    by_id2 = {x["id"]: x for x in out2["models"]}
    assert by_id2["m1"]["backend"] == "remote"
    assert by_id2["m1"]["load_mode"] == "unknown"


def test_list_models_payload_single_backend(monkeypatch):
    req = _req()
    req.app.state.model_loaded = True
    backend = _Backend("m-single", "transformers")
    monkeypatch.setattr(ml, "settings_from_request", lambda request: SimpleNamespace(model_id="m-fallback"), raising=True)
    monkeypatch.setattr(ml, "deployment_capabilities", lambda request: {}, raising=True)
    monkeypatch.setattr(ml, "effective_capabilities", lambda mid, request=None: {"generate": True}, raising=True)

    monkeypatch.setattr(ml, "effective_model_load_mode_from_request", lambda request: "eager", raising=True)
    out = ml.list_models_payload(request=req, llm=backend)
    assert out["models"][0]["loaded"] is True

    monkeypatch.setattr(ml, "effective_model_load_mode_from_request", lambda request: "lazy", raising=True)
    out2 = ml.list_models_payload(request=req, llm=backend)
    assert out2["models"][0]["loaded"] is None
