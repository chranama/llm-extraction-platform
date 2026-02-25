from __future__ import annotations

from types import SimpleNamespace

from llm_server.services.limits import config as cfg


def test_primitives_and_env(monkeypatch):
    assert cfg._truthy(True) is True
    assert cfg._truthy("on") is True
    assert cfg._truthy("off") is False
    assert cfg._as_int("3", 1) == 3
    assert cfg._as_int("x", 1) == 1
    assert cfg._as_float("1.5", 0.5) == 1.5
    assert cfg._as_float("x", 0.5) == 0.5

    d = {"a": {"b": 2}}
    assert cfg._get_attr_or_key(d, "a") == {"b": 2}
    assert cfg._get_nested(d, "a.b") == 2
    assert cfg._get_nested(d, "a.missing") is None

    monkeypatch.setenv("X_ENV", "  v ")
    assert cfg._env("X_ENV") == "v"
    monkeypatch.setenv("X_ENV", " ")
    assert cfg._env("X_ENV") is None


def test_load_generate_gate_config_from_settings_and_env(monkeypatch):
    settings = SimpleNamespace(
        limits={
            "generate_gate": {
                "enabled": False,
                "max_concurrent": 0,
                "max_queue": -1,
                "timeout_seconds": 0.1,
                "fail_fast": False,
                "count_queued_as_in_flight": True,
            }
        }
    )
    out1 = cfg.load_generate_gate_config(settings)
    assert out1.enabled is False
    assert out1.max_concurrent == 1
    assert out1.max_queue == 0
    assert out1.timeout_seconds == 0.5
    assert out1.fail_fast is False
    assert out1.count_queued_as_in_flight is True

    monkeypatch.setenv("GENERATE_GATE_ENABLED", "1")
    monkeypatch.setenv("MAX_CONCURRENT_GENERATIONS", "7")
    monkeypatch.setenv("MAX_GENERATE_QUEUE", "9")
    monkeypatch.setenv("GENERATE_TIMEOUT_S", "3.2")
    monkeypatch.setenv("GENERATE_GATE_FAIL_FAST", "1")
    monkeypatch.setenv("GENERATE_GATE_COUNT_QUEUED_AS_IN_FLIGHT", "0")
    out2 = cfg.load_generate_gate_config(settings)
    assert out2.enabled is True
    assert out2.max_concurrent == 7
    assert out2.max_queue == 9
    assert out2.timeout_seconds == 3.2
    assert out2.fail_fast is True
    assert out2.count_queued_as_in_flight is False


def test_load_generate_early_reject_config_from_settings_and_env(monkeypatch):
    settings = SimpleNamespace(
        limits=SimpleNamespace(
            generate_early_reject={
                "enabled": True,
                "reject_queue_depth_gte": -1,
                "reject_in_flight_gte": -2,
                "routes": ["/a", " ", "/b"],
            }
        )
    )
    out1 = cfg.load_generate_early_reject_config(settings)
    assert out1.enabled is True
    assert out1.reject_queue_depth_gte == 0
    assert out1.reject_in_flight_gte == 0
    assert out1.routes == ("/a", "/b")

    monkeypatch.setenv("GENERATE_EARLY_REJECT_ENABLED", "0")
    monkeypatch.setenv("GENERATE_EARLY_REJECT_QUEUE_DEPTH_GTE", "5")
    monkeypatch.setenv("GENERATE_EARLY_REJECT_IN_FLIGHT_GTE", "7")
    monkeypatch.setenv("GENERATE_EARLY_REJECT_ROUTES", "/x, /y")
    out2 = cfg.load_generate_early_reject_config(settings)
    assert out2.enabled is False
    assert out2.reject_queue_depth_gte == 5
    assert out2.reject_in_flight_gte == 7
    assert out2.routes == ("/x", "/y")
