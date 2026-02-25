from __future__ import annotations

from types import SimpleNamespace

import pytest

from llm_server.core.errors import AppError
from llm_server.services.api_deps.routing import models as routing_models
from llm_server.services.llm_runtime.llm_registry import MultiModelManager


class _Backend:
    def __init__(self, model_id: str):
        self.model_id = model_id


class _Req:
    def __init__(
        self, settings, llm=None, *, runtime_default_model_id=None, loaded_model_id=None
    ):
        self.app = SimpleNamespace(
            state=SimpleNamespace(
                settings=settings,
                llm=llm,
                runtime_default_model_id=runtime_default_model_id,
                loaded_model_id=loaded_model_id,
            )
        )


def _settings(*, model_id: str, allowed_models: list[str] | None = None):
    return SimpleNamespace(
        model_id=model_id, allowed_models=allowed_models or [], all_model_ids=[]
    )


def test_resolve_model_multimodel_override_missing():
    llm = MultiModelManager(models={"m1": _Backend("m1")}, default_id="m1")
    req = _Req(_settings(model_id="m1", allowed_models=["m1"]), llm)

    with pytest.raises(AppError) as e:
        routing_models.resolve_model(llm, "nope", request=req)

    assert e.value.code == "model_missing"
    assert e.value.status_code == 500


def test_resolve_model_multimodel_no_override_uses_default_id():
    llm = MultiModelManager(
        models={"m1": _Backend("m1"), "m2": _Backend("m2")}, default_id="m2"
    )
    req = _Req(_settings(model_id="m2", allowed_models=["m1", "m2"]), llm)

    mid, _ = routing_models.resolve_model(llm, None, capability="generate", request=req)
    assert mid == "m2"


def test_resolve_model_multimodel_extract_no_override_uses_default_for_capability():
    llm = MultiModelManager(
        models={"gen": _Backend("gen"), "ext": _Backend("ext")},
        default_id="gen",
        model_meta={
            "gen": {"capabilities": ["generate"]},
            "ext": {"capabilities": ["extract"]},
        },
    )
    req = _Req(_settings(model_id="gen", allowed_models=["gen", "ext"]), llm)

    mid, _ = routing_models.resolve_model(llm, None, capability="extract", request=req)
    assert mid == "ext"


def test_resolve_model_multimodel_allowed_models_excludes_chosen():
    llm = MultiModelManager(models={"m1": _Backend("m1")}, default_id="m1")
    req = _Req(_settings(model_id="m1", allowed_models=["other"]), llm)

    with pytest.raises(AppError) as e:
        routing_models.resolve_model(llm, None, request=req)

    assert e.value.code == "model_not_allowed"
    assert e.value.status_code == 400


def test_resolve_model_dict_registry_override_missing():
    llm = {"a": _Backend("a")}
    req = _Req(_settings(model_id="a", allowed_models=["a", "b"]), llm)

    with pytest.raises(AppError) as e:
        routing_models.resolve_model(llm, "b", request=req)

    assert e.value.code == "model_missing"
    assert e.value.status_code == 500


def test_resolve_model_dict_registry_override_not_allowed():
    llm = {"a": _Backend("a")}
    req = _Req(_settings(model_id="a", allowed_models=["a"]), llm)

    with pytest.raises(AppError) as e:
        routing_models.resolve_model(llm, "b", request=req)

    assert e.value.code == "model_not_allowed"
    assert e.value.status_code == 400


def test_resolve_model_dict_registry_default_fallback_order():
    llm = {"a": _Backend("a"), "b": _Backend("b")}
    req = _Req(_settings(model_id="b", allowed_models=["a", "b"]), llm)

    mid, _ = routing_models.resolve_model(llm, None, request=req)
    assert mid == "b"


def test_resolve_model_single_backend_override_allowed_and_not_allowed():
    llm = _Backend("m1")
    req = _Req(_settings(model_id="m1", allowed_models=["m1"]), llm)

    mid, _ = routing_models.resolve_model(llm, "m1", request=req)
    assert mid == "m1"

    with pytest.raises(AppError) as e:
        routing_models.resolve_model(llm, "m2", request=req)

    assert e.value.code == "model_not_allowed"
    assert e.value.status_code == 400
