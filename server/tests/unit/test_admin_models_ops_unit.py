from __future__ import annotations

from types import SimpleNamespace

from llm_server.services.api_deps.admin import models_ops as mo
from llm_server.services.llm_runtime.llm_registry import MultiModelManager


class _Backend:
    def __init__(self, model_id: str):
        self.model_id = model_id


def test_allowed_model_ids_from_settings_prefers_allowed_over_legacy():
    s1 = SimpleNamespace(allowed_models=["a", "  ", "b"], all_model_ids=["x"])
    assert mo.allowed_model_ids_from_settings(s1) == ["a", "b"]

    s2 = SimpleNamespace(allowed_models=[], all_model_ids=["x", "", "y"])
    assert mo.allowed_model_ids_from_settings(s2) == ["x", "y"]


def test_summarize_registry_for_multimodel_manager():
    llm = MultiModelManager(models={"m1": _Backend("m1"), "m2": _Backend("m2")}, default_id="m2")
    default_model, ids = mo.summarize_registry(llm, fallback_default="fb")
    assert default_model == "m2"
    assert ids == ["m1", "m2"]


def test_summarize_registry_for_duck_typed_registry_and_single_backend():
    duck = SimpleNamespace(default_id="d1", models={"d1": object(), "d2": object()})
    d0, ids0 = mo.summarize_registry(duck, fallback_default="fb")
    assert (d0, ids0) == ("d1", ["d1", "d2"])

    single = SimpleNamespace(model_id="only")
    d1, ids1 = mo.summarize_registry(single, fallback_default="fb")
    assert (d1, ids1) == ("only", ["only"])

    missing = SimpleNamespace()
    d2, ids2 = mo.summarize_registry(missing, fallback_default="fb")
    assert (d2, ids2) == ("fb", ["fb"])


def test_runtime_default_model_id_from_app_and_get_loader(monkeypatch):
    req = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(runtime_default_model_id="  mm  ")))
    assert mo.runtime_default_model_id_from_app(req) == "mm"

    loader = object()
    monkeypatch.setattr(mo, "get_runtime_model_loader", lambda request: loader, raising=True)
    assert mo.get_loader(req) is loader
