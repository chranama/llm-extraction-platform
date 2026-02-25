from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_server.core.errors import AppError
from llm_server.services.llm_runtime import llm_config as cfg


def test_expand_env_in_str_and_nested_objects(monkeypatch):
    monkeypatch.setenv("X_NAME", "alice")
    s = cfg._expand_env_in_str("hello-${X_NAME}-${MISSING:-fallback}")
    assert s == "hello-alice-fallback"

    obj = {"a": "${X_NAME}", "b": ["${MISSING:-z}"], "c": ("${X_NAME}",)}
    out = cfg._expand_env(obj)
    assert out["a"] == "alice"
    assert out["b"] == ["z"]
    assert out["c"] == ("alice",)


def test_merge_models_list_by_id_and_deep_merge():
    base = [{"id": "m1", "remote": {"a": 1}}, {"id": "m2"}, {"noid": True}]
    overlay = [{"id": "m1", "remote": {"b": 2}}, {"id": "m3"}]
    merged = cfg._merge_models_list_by_id(base, overlay)
    assert merged[0]["id"] == "m1"
    assert merged[0]["remote"] == {"a": 1, "b": 2}
    assert merged[1]["id"] == "m2"
    assert merged[2]["id"] == "m3"
    assert merged[3] == {"noid": True}


def test_select_profile_with_alias_and_fallback(monkeypatch):
    raw = {
        "base": {"models": [{"id": "m1"}], "defaults": {"load_mode": "lazy"}},
        "profiles": {
            "host-transformers": {"defaults": {"backend": "transformers"}},
            "host-llama": {"defaults": {"backend": "llamacpp"}},
        },
    }

    monkeypatch.setenv("APP_PROFILE", "host")
    monkeypatch.delenv("MODELS_PROFILE", raising=False)
    out1 = cfg._select_profile(raw)
    assert out1["_selected_profile_used"] == "host-transformers"
    assert out1["defaults"]["backend"] == "transformers"

    monkeypatch.setenv("MODELS_PROFILE", "unknown")
    out2 = cfg._select_profile(raw)
    assert out2["_selected_profile_requested"] == "unknown"
    assert out2["_selected_profile_used"] == "host-transformers"


def test_validate_enum_and_backend_block_errors():
    assert cfg._validate_enum("ON", field="f", path="p", allowed={"on", "off"}) == "on"
    with pytest.raises(AppError):
        cfg._validate_enum("bad", field="f", path="p", allowed={"on"})

    assert cfg._normalize_backend_block({"k": 1}, path="p", field="f") == {"k": 1}
    with pytest.raises(AppError):
        cfg._normalize_backend_block("bad", path="p", field="f")


def test_readiness_and_deployment_key_normalization():
    assert cfg._normalize_readiness_mode("probe", path="p", field="f") == "probe"
    assert cfg._normalize_deployment_key("  key1 ", path="p", field="f") == "key1"
    assert cfg._normalize_deployment_key("  ", path="p", field="f") is None

    with pytest.raises(AppError):
        cfg._normalize_readiness_mode("bad", path="p", field="f")
    with pytest.raises(AppError):
        cfg._normalize_deployment_key(123, path="p", field="f")


def test_capability_validation_and_bool_conversion():
    raw = {"generate": {"enabled": True, "x": 1}, "extract": False}
    valid = cfg._validate_capabilities_mapping(raw, path="p", field="caps")
    assert valid == raw

    with pytest.raises(AppError):
        cfg._validate_capabilities_mapping({"bad": True}, path="p", field="caps")
    with pytest.raises(AppError):
        cfg._cap_value_to_bool({"enabled": "yes"}, path="p", field="caps", key="generate")

    assert cfg._cap_value_to_bool(True, path="p", field="caps", key="generate") is True
    assert cfg._capabilities_raw_to_boolmap(raw, path="p", field="caps") == {"generate": True, "extract": False}


def test_merge_capabilities_raw_and_deep_merge():
    defaults = {"generate": {"enabled": True, "meta": {"a": 1}}, "extract": False}
    declared = {"generate": {"enabled": False, "meta": {"b": 2}}}
    merged = cfg._merge_capabilities_raw(defaults_caps=defaults, declared_caps=declared)
    assert merged["extract"] is False
    assert merged["generate"]["enabled"] is False
    assert merged["generate"]["meta"] == {"a": 1, "b": 2}


def test_normalize_model_entry_string_and_dict_backcompat():
    defaults = {
        "backend": "transformers",
        "load_mode": "lazy",
        "capabilities": {"generate": True},
        "deployment_key": "dk-default",
        "readiness_mode": "generate",
        "deployment": {"tier": "base"},
        "assessment": {},
        "backend_constraints": {},
        "transformers": {"dtype": "float16"},
        "llamacpp": {},
        "remote": {"timeout_seconds": 10},
    }
    s = cfg._normalize_model_entry("m1", path="p", defaults=defaults)
    assert s.id == "m1"
    assert s.backend == "transformers"
    assert s.capabilities == {"generate": True}

    d = cfg._normalize_model_entry(
        {
            "id": "m2",
            "backend": "remote",
            "remote_base_url": "http://x",
            "remote_model_id": "rm",
            "capabilities": {"generate": {"enabled": True}, "extract": {"enabled": False}},
            "deployment": {"tier": "override"},
            "notes": " n ",
        },
        path="p",
        defaults=defaults,
    )
    assert d.id == "m2"
    assert d.backend == "remote"
    assert d.remote["base_url"] == "http://x"
    assert d.remote["model_name"] == "rm"
    assert d.deployment["tier"] == "override"
    assert d.notes == "n"


def test_apply_effective_service_caps_and_deployment_key_validation(monkeypatch):
    monkeypatch.delenv("ENABLE_GENERATE", raising=False)
    monkeypatch.delenv("ENABLE_EXTRACT", raising=False)

    s = SimpleNamespace(enable_generate=True, enable_extract=True)
    primary = cfg.ModelSpec(id="m1", capabilities={"generate": True, "extract": False})
    gen, ext = cfg._apply_effective_service_caps_from_primary(s=s, primary=primary, defaults_caps_bool={"extract": True})
    assert (gen, ext) == (True, False)
    assert s.enable_extract is False

    assert cfg._should_enforce_deployment_keys(selected_profile="prod") is True
    monkeypatch.setenv("ALLOW_GENERIC_DEPLOYMENT_KEY", "1")
    assert cfg._should_enforce_deployment_keys(selected_profile="prod") is False

    monkeypatch.delenv("ALLOW_GENERIC_DEPLOYMENT_KEY", raising=False)
    with pytest.raises(AppError):
        cfg._validate_deployment_keys_or_raise(
            specs=[cfg.ModelSpec(id="m1", deployment_key="default"), cfg.ModelSpec(id="m2", deployment_key=None)],
            path="models.yaml",
            selected_profile="prod",
        )


def test_load_yaml_happy_path_and_failures(tmp_path: Path):
    p = tmp_path / "models.yaml"
    p.write_text("default_model: m1\nmodels:\n  - id: m1\n", encoding="utf-8")
    out = cfg._load_yaml(str(p))
    assert out["default_model"] == "m1"

    with pytest.raises(AppError):
        cfg._load_yaml(str(tmp_path / "missing.yaml"))
