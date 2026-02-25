from __future__ import annotations

from types import SimpleNamespace

import pytest

from llm_server.services.api_deps.routing import models as rm
from llm_server.services.llm_runtime.llm_registry import MultiModelManager


class _Backend:
    def __init__(self, model_id: str, backend_name: str = "transformers"):
        self.model_id = model_id
        self.backend_name = backend_name


def _req(settings=None, llm=None):
    return SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                settings=settings or SimpleNamespace(model_id="m1", allowed_models=[], all_model_ids=[]),
                llm=llm,
                model_loaded=False,
                model_error=None,
                loaded_model_id=None,
                runtime_default_model_id=None,
            )
        )
    )


def test_allowed_and_default_helpers(monkeypatch):
    req1 = _req(settings=SimpleNamespace(model_id="  m1 ", allowed_models=["a", " ", "b"], all_model_ids=["x"]))
    monkeypatch.setattr(rm, "settings_from_request", lambda request: request.app.state.settings, raising=True)
    assert rm.allowed_model_ids(request=req1) == ["a", "b"]
    assert rm.default_model_id_from_settings(request=req1) == "m1"

    req2 = _req(settings=SimpleNamespace(model_id="", allowed_models=[], all_model_ids=["x", "", "y"]))
    assert rm.allowed_model_ids(request=req2) == ["x", "y"]
    assert rm.default_model_id_from_settings(request=req2) == ""


def test_runtime_default_and_loaded_model_id_fallback(monkeypatch):
    req = _req()
    req.app.state.runtime_default_model_id = "  rt-mid "
    req.app.state.loaded_model_id = "  ld-mid "

    class _MS:
        def __init__(self, _state):
            pass

        @staticmethod
        def snapshot():
            raise RuntimeError("no model_state")

    monkeypatch.setattr(rm, "ModelStateStore", _MS, raising=True)
    assert rm.runtime_default_model_id(req) == "rt-mid"
    assert rm.loaded_model_id(req) == "ld-mid"


def test_resolve_default_model_id_and_backend_obj_branches(monkeypatch):
    # Multi model with runtime default present in registry
    mm = MultiModelManager(models={"m1": _Backend("m1", "remote"), "m2": _Backend("m2", "llamacpp")}, default_id="m2")
    req1 = _req(llm=mm)
    req1.app.state.runtime_default_model_id = "m1"

    class _MS1:
        def __init__(self, state):
            self.state = state

        def snapshot(self):
            return SimpleNamespace(
                runtime_default_model_id=self.state.runtime_default_model_id,
                loaded_model_id=self.state.loaded_model_id,
                model_loaded=self.state.model_loaded,
                model_error=self.state.model_error,
            )

    monkeypatch.setattr(rm, "ModelStateStore", _MS1, raising=True)
    mid, bname, _ = rm.resolve_default_model_id_and_backend_obj(req1)
    assert (mid, bname) == ("m1", "remote")

    # fallback to default when runtime default missing
    req1.app.state.runtime_default_model_id = "missing"
    mid2, bname2, _ = rm.resolve_default_model_id_and_backend_obj(req1)
    assert (mid2, bname2) == ("m2", "llamacpp")

    # single backend uses loaded model id first
    req2 = _req(llm=_Backend("cfg-mid", "transformers"))
    req2.app.state.loaded_model_id = "loaded-mid"
    mid3, bname3, _ = rm.resolve_default_model_id_and_backend_obj(req2)
    assert (mid3, bname3) == ("loaded-mid", "transformers")


def test_model_flags_fallback_when_model_state_errors(monkeypatch):
    req = _req()
    req.app.state.model_loaded = True
    req.app.state.model_error = "  bad "
    req.app.state.loaded_model_id = "  m1 "
    req.app.state.runtime_default_model_id = "  m2 "

    class _MS:
        def __init__(self, _state):
            pass

        @staticmethod
        def snapshot():
            raise RuntimeError("no state")

    monkeypatch.setattr(rm, "ModelStateStore", _MS, raising=True)
    vals = rm.model_flags_from_app_state(req)
    assert vals == (True, "bad", "m1", "m2")


def test_settings_and_per_model_readiness_mode(monkeypatch):
    req = _req(settings=SimpleNamespace(model_readiness_mode="PROBE", model_id="m1", allowed_models=[], all_model_ids=[]))
    monkeypatch.setattr(rm, "settings_from_request", lambda request: request.app.state.settings, raising=True)
    assert rm.settings_readiness_mode(req) == "probe"

    req2 = _req(settings=SimpleNamespace(model_readiness_mode="weird", model_id="m1", allowed_models=[], all_model_ids=[]))
    assert rm.settings_readiness_mode(req2) == "generate"

    mm = MultiModelManager(models={"m1": _Backend("m1")}, default_id="m1", model_meta={"m1": {"readiness_mode": "off"}})
    req3 = _req(settings=SimpleNamespace(model_readiness_mode="generate", model_id="m1", allowed_models=[], all_model_ids=[]), llm=mm)
    assert rm.per_model_readiness_mode(req3, model_id="m1") == "off"
    assert rm.per_model_readiness_mode(req3, model_id=None) == "generate"
