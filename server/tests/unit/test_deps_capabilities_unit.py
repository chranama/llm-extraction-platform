from __future__ import annotations

from types import SimpleNamespace

import pytest

from llm_server.core.errors import AppError
from llm_server.services.api_deps.enforcement import capabilities as caps


class FakeModelsConfig:
    def __init__(self, defaults: dict, models: list):
        self.defaults = defaults
        self.models = models


class FakeModelSpec:
    def __init__(self, model_id: str, capabilities=None):
        self.id = model_id
        self.capabilities = capabilities


class _Req:
    def __init__(self, *, enable_extract: bool = True, enable_generate: bool = True):
        settings = SimpleNamespace(
            enable_extract=enable_extract, enable_generate=enable_generate
        )
        self.app = SimpleNamespace(state=SimpleNamespace(settings=settings, llm=None))


def test_effective_capabilities_policy_disables_extract(
    monkeypatch: pytest.MonkeyPatch,
):
    req = _Req(enable_extract=True, enable_generate=True)

    monkeypatch.setattr(
        caps,
        "_model_capabilities_from_models_yaml",
        lambda _mid, request=None: {"extract": True, "generate": True},
    )
    monkeypatch.setattr(
        caps, "policy_capability_overrides", lambda _mid, request: {"extract": False}
    )

    out = caps.effective_capabilities("m1", request=req)
    assert out["extract"] is False
    assert out["generate"] is True


def test_require_capability_policy_denies_extract(monkeypatch: pytest.MonkeyPatch):
    req = _Req(enable_extract=True, enable_generate=True)

    monkeypatch.setattr(
        caps,
        "_model_capabilities_from_models_yaml",
        lambda _mid, request=None: {"extract": True, "generate": True},
    )
    monkeypatch.setattr(
        caps, "policy_capability_overrides", lambda _mid, request: {"extract": False}
    )

    with pytest.raises(AppError) as ei:
        caps.require_capability("m1", "extract", request=req)

    e = ei.value
    assert e.code == "capability_not_supported"
    assert e.status_code == 400
    assert e.extra and e.extra.get("model_id") == "m1"


def test_deployment_gate_still_wins_over_policy(monkeypatch: pytest.MonkeyPatch):
    req = _Req(enable_extract=False, enable_generate=True)

    monkeypatch.setattr(
        caps,
        "_model_capabilities_from_models_yaml",
        lambda _mid, request=None: {"extract": True, "generate": True},
    )
    monkeypatch.setattr(
        caps, "policy_capability_overrides", lambda _mid, request: {"extract": True}
    )

    with pytest.raises(AppError) as ei:
        caps.require_capability("m1", "extract", request=req)

    e = ei.value
    assert e.code == "capability_disabled"
    assert e.status_code == 501


def test_models_yaml_unspecified_means_allow_all(monkeypatch: pytest.MonkeyPatch):
    cfg = FakeModelsConfig(
        defaults={}, models=[FakeModelSpec(model_id="m1", capabilities=None)]
    )
    monkeypatch.setattr(caps, "models_config_from_request", lambda _req=None: cfg)

    assert caps.model_capabilities("m1", request=None) is None

    caps.require_capability("m1", "extract", request=None)
    caps.require_capability("m1", "generate", request=None)


def test_defaults_missing_key_defaults_true(monkeypatch: pytest.MonkeyPatch):
    cfg = FakeModelsConfig(
        defaults={"capabilities": {"extract": False}},
        models=[FakeModelSpec(model_id="m1", capabilities=None)],
    )
    monkeypatch.setattr(caps, "models_config_from_request", lambda _req=None: cfg)

    out = caps.effective_capabilities("m1", request=None)
    assert out["extract"] is False
    assert out["generate"] is True


def test_model_overrides_defaults(monkeypatch: pytest.MonkeyPatch):
    cfg = FakeModelsConfig(
        defaults={"capabilities": {"extract": True}},
        models=[FakeModelSpec(model_id="m1", capabilities={"extract": False})],
    )
    monkeypatch.setattr(caps, "models_config_from_request", lambda _req=None: cfg)

    mc = caps.model_capabilities("m1", request=None)
    assert mc is not None
    assert mc["extract"] is False

    with pytest.raises(AppError) as e:
        caps.require_capability("m1", "extract", request=None)

    assert e.value.code == "capability_not_supported"
    assert e.value.status_code == 400


def test_deployment_disables_capability_overrides_model(
    monkeypatch: pytest.MonkeyPatch,
):
    cfg = FakeModelsConfig(
        defaults={"capabilities": {"extract": True}},
        models=[FakeModelSpec(model_id="m1", capabilities={"extract": True})],
    )
    monkeypatch.setattr(caps, "models_config_from_request", lambda _req=None: cfg)
    monkeypatch.setattr(
        caps,
        "deployment_capabilities",
        lambda request=None: {"generate": True, "extract": False},
    )

    with pytest.raises(AppError) as e:
        caps.require_capability("m1", "extract", request=None)

    assert e.value.code == "capability_disabled"
    assert e.value.status_code == 501
