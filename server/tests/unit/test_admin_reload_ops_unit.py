from __future__ import annotations

from types import SimpleNamespace

import pytest

from llm_server.services.api_deps.admin import reload_ops as ro


class _Loader:
    def __init__(self, *, llm_obj=None, refresh_raises: Exception | None = None, rebuild_raises: Exception | None = None):
        self.llm_obj = llm_obj if llm_obj is not None else SimpleNamespace(model_id="m-new")
        self.refresh_raises = refresh_raises
        self.rebuild_raises = rebuild_raises
        self.refreshed = False
        self.rebuilt = False

    async def refresh_models_config(self):
        self.refreshed = True
        if self.refresh_raises:
            raise self.refresh_raises

    async def rebuild_llm_registry(self):
        self.rebuilt = True
        if self.rebuild_raises:
            raise self.rebuild_raises
        return self.llm_obj


def _request():
    state = SimpleNamespace(
        policy_snapshot="old",
        model_error="old_err",
        model_loaded=True,
        loaded_model_id="old_mid",
        runtime_default_model_id="runtime-mid",
        llm=None,
    )
    return SimpleNamespace(app=SimpleNamespace(state=state))


@pytest.mark.anyio
async def test_reload_runtime_state_happy_path(monkeypatch):
    req = _request()
    loader = _Loader(llm_obj=SimpleNamespace(model_id="m-final"))
    snap = SimpleNamespace(ok=True, model_id="m-final", enable_extract=True, source_path="/tmp/p.json", error=None)

    monkeypatch.setattr(ro, "get_settings", lambda: SimpleNamespace(model_id="fallback"), raising=True)
    monkeypatch.setattr(ro, "clear_models_config_cache", lambda: None, raising=True)
    monkeypatch.setattr(ro, "reload_policy_snapshot", lambda request: snap, raising=True)
    monkeypatch.setattr(ro, "effective_capabilities", lambda mid, request=None: {"extract": True}, raising=True)
    monkeypatch.setattr(ro, "snapshot_generate_cap", lambda s: 123, raising=True)
    monkeypatch.setattr(ro, "summarize_registry", lambda llm, fallback_default: ("m-final", ["m-final"]), raising=True)

    out, snap_out = await ro.reload_runtime_state(request=req, loader=loader)
    assert snap_out is snap
    assert loader.refreshed is True
    assert loader.rebuilt is True
    assert req.app.state.model_loaded is False
    assert req.app.state.loaded_model_id is None
    assert out["models"]["default_model"] == "m-final"
    assert out["models"]["runtime_default_model"] == "runtime-mid"
    assert out["policy"]["generate_max_new_tokens_cap"] == 123
    assert out["effective"]["extract_enabled"] is True


@pytest.mark.anyio
async def test_reload_runtime_state_continues_when_cache_or_refresh_fail(monkeypatch):
    req = _request()
    loader = _Loader(refresh_raises=RuntimeError("refresh fail"))
    snap = SimpleNamespace(ok=False, model_id=None, enable_extract=False, source_path=None, error="bad")

    def _cache_clear():
        raise RuntimeError("cache fail")

    monkeypatch.setattr(ro, "get_settings", lambda: SimpleNamespace(model_id="fallback"), raising=True)
    monkeypatch.setattr(ro, "clear_models_config_cache", _cache_clear, raising=True)
    monkeypatch.setattr(ro, "reload_policy_snapshot", lambda request: snap, raising=True)
    monkeypatch.setattr(ro, "effective_capabilities", lambda mid, request=None: {"extract": False}, raising=True)
    monkeypatch.setattr(ro, "snapshot_generate_cap", lambda s: None, raising=True)
    monkeypatch.setattr(ro, "summarize_registry", lambda llm, fallback_default: ("fallback", ["fallback"]), raising=True)

    out, _ = await ro.reload_runtime_state(request=req, loader=loader)
    assert loader.rebuilt is True
    assert out["effective"]["extract_enabled"] is False


@pytest.mark.anyio
async def test_reload_runtime_state_rebuild_failure_sets_model_error(monkeypatch):
    req = _request()
    boom = RuntimeError("rebuild fail")
    loader = _Loader(rebuild_raises=boom)

    monkeypatch.setattr(ro, "get_settings", lambda: SimpleNamespace(model_id="fallback"), raising=True)
    monkeypatch.setattr(ro, "clear_models_config_cache", lambda: None, raising=True)

    with pytest.raises(RuntimeError):
        await ro.reload_runtime_state(request=req, loader=loader)
    assert "rebuild fail" in req.app.state.model_error


@pytest.mark.anyio
async def test_reload_runtime_state_effective_capabilities_error_defaults_false(monkeypatch):
    req = _request()
    loader = _Loader()
    snap = SimpleNamespace(ok=True, model_id="m1", enable_extract=True, source_path=None, error=None)

    monkeypatch.setattr(ro, "get_settings", lambda: SimpleNamespace(model_id="fallback"), raising=True)
    monkeypatch.setattr(ro, "clear_models_config_cache", lambda: None, raising=True)
    monkeypatch.setattr(ro, "reload_policy_snapshot", lambda request: snap, raising=True)
    monkeypatch.setattr(ro, "snapshot_generate_cap", lambda s: None, raising=True)
    monkeypatch.setattr(ro, "summarize_registry", lambda llm, fallback_default: ("m1", ["m1"]), raising=True)

    def _caps(*args, **kwargs):
        raise RuntimeError("caps failed")

    monkeypatch.setattr(ro, "effective_capabilities", _caps, raising=True)

    out, _ = await ro.reload_runtime_state(request=req, loader=loader)
    assert out["effective"]["extract_enabled"] is False
