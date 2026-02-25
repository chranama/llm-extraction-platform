from __future__ import annotations

from types import SimpleNamespace

import pytest

from llm_server.core.errors import AppError
from llm_server.services.llm_runtime import llm_loader as mod
from llm_server.services.llm_runtime.llm_registry import MultiModelManager


class _Metric:
    def labels(self, **kwargs):
        return self

    def inc(self):
        return None

    def observe(self, value):
        return None


class _Backend:
    def __init__(self, *, model_id: str, backend_name: str = "transformers", loaded: bool = False):
        self.model_id = model_id
        self.backend_name = backend_name
        self._loaded = loaded
        self._probe = None

    def is_loaded(self):
        return self._loaded

    def ensure_loaded(self):
        self._loaded = True

    def probe(self):
        if callable(self._probe):
            return self._probe()
        return self._probe


def _mk_loader():
    state = SimpleNamespace(
        llm=None,
        models_config=None,
        model_error=None,
        model_load_mode="lazy",
        model_loaded=False,
        loaded_model_id=None,
        runtime_default_model_id=None,
    )
    return mod.RuntimeModelLoader(state), state


@pytest.fixture(autouse=True)
def _patch_metrics(monkeypatch):
    m = _Metric()
    monkeypatch.setattr(mod, "LLM_LOADER_OPS_TOTAL", m, raising=True)
    monkeypatch.setattr(mod, "LLM_LOADER_OPS_FAIL_TOTAL", m, raising=True)
    monkeypatch.setattr(mod, "LLM_LOADER_OP_LATENCY_SECONDS", m, raising=True)
    monkeypatch.setattr(mod, "set_state_gauges", lambda **kwargs: None, raising=True)


@pytest.mark.anyio
async def test_refresh_and_rebuild(monkeypatch):
    ldr, state = _mk_loader()
    cfg = SimpleNamespace(primary_id="m1")
    llm = _Backend(model_id="m1")
    monkeypatch.setattr(mod, "load_models_config", lambda: cfg, raising=True)
    monkeypatch.setattr(mod, "build_llm_from_settings", lambda: llm, raising=True)

    out_cfg = await ldr.refresh_models_config()
    out_llm = await ldr.rebuild_llm_registry()
    assert out_cfg is cfg
    assert out_llm is llm
    assert state.models_config is cfg
    assert state.llm is llm


@pytest.mark.anyio
async def test_load_model_invalid_request():
    ldr, _ = _mk_loader()
    with pytest.raises(AppError) as e:
        await ldr.load_model("")
    assert e.value.code == "invalid_request"


@pytest.mark.anyio
async def test_load_model_external_backend_noop(monkeypatch):
    ldr, state = _mk_loader()
    ext = _Backend(model_id="m-ext", backend_name="remote")
    monkeypatch.setattr(mod, "build_llm_from_settings", lambda: ext, raising=True)

    res = await ldr.load_model("m-ext")
    assert res.loaded is False
    assert res.detail["status"] == "noop_external"
    assert state.loaded_model_id == "m-ext"
    assert state.model_loaded is False


@pytest.mark.anyio
async def test_load_model_transformers_loads(monkeypatch):
    ldr, state = _mk_loader()
    b = _Backend(model_id="m1", backend_name="transformers", loaded=False)
    monkeypatch.setattr(mod, "build_llm_from_settings", lambda: b, raising=True)
    monkeypatch.setattr(ldr, "_ensure_loaded_async", lambda backend: mod.anyio.sleep(0), raising=True)

    # ensure_loaded_async above does not mutate backend; set force path via sync helper
    monkeypatch.setattr(ldr, "_ensure_loaded_async", lambda backend: mod.anyio.to_thread.run_sync(backend.ensure_loaded), raising=True)
    res = await ldr.load_model("m1")
    assert res.loaded is True
    assert res.detail["status"] == "loaded_in_process"
    assert state.loaded_model_id == "m1"
    assert state.model_loaded is True


@pytest.mark.anyio
async def test_load_model_wraps_unexpected_exception(monkeypatch):
    ldr, state = _mk_loader()
    b = _Backend(model_id="m1", backend_name="transformers", loaded=False)
    monkeypatch.setattr(mod, "build_llm_from_settings", lambda: b, raising=True)

    async def _boom(_backend):
        raise RuntimeError("explode")

    monkeypatch.setattr(ldr, "_ensure_loaded_async", _boom, raising=True)
    with pytest.raises(AppError) as e:
        await ldr.load_model("m1")
    assert e.value.code == "model_load_failed"
    assert state.model_loaded is False
    assert state.loaded_model_id is None
    assert "RuntimeError: explode" in (state.model_error or "")


@pytest.mark.anyio
async def test_load_default_prefers_runtime_default_then_registry_default(monkeypatch):
    ldr, state = _mk_loader()
    mm = MultiModelManager(
        models={"m1": _Backend(model_id="m1"), "m2": _Backend(model_id="m2")},
        default_id="m2",
    )
    state.llm = mm
    state.runtime_default_model_id = "m1"

    calls = []

    async def _load_model(mid, force=False):
        calls.append((mid, force))
        return mod.LoadResult(model_id=mid, loaded=True, load_mode="lazy", detail={})

    monkeypatch.setattr(ldr, "load_model", _load_model, raising=True)
    out1 = await ldr.load_default(force=True)
    assert out1.model_id == "m1"
    assert calls[-1] == ("m1", True)

    ldr._ms.set_runtime_default_model_id(None)
    out2 = await ldr.load_default(force=False)
    assert out2.model_id == "m2"


@pytest.mark.anyio
async def test_set_default_and_clear_runtime_default(monkeypatch):
    ldr, state = _mk_loader()
    state.llm = MultiModelManager(models={"m1": _Backend(model_id="m1")}, default_id="m1")

    out = await ldr.set_default_model("m1")
    assert out["default_model"] == "m1"
    assert state.runtime_default_model_id == "m1"

    cleared = await ldr.clear_runtime_default()
    assert cleared["default_model"] is None
    assert state.runtime_default_model_id is None

    with pytest.raises(AppError):
        await ldr.set_default_model("")
    with pytest.raises(AppError) as e:
        await ldr.set_default_model("missing")
    assert e.value.code == "model_missing"


@pytest.mark.anyio
async def test_probe_model_paths(monkeypatch):
    ldr, state = _mk_loader()
    b = _Backend(model_id="m1", backend_name="transformers", loaded=True)
    state.llm = b

    b._probe = {"ok": False, "note": "x"}
    r1 = await ldr.probe_model("m1")
    assert r1.ok is False
    assert r1.detail["note"] == "x"

    def _probe_boom():
        raise RuntimeError("no probe")

    b._probe = _probe_boom
    r2 = await ldr.probe_model("m1")
    assert r2.ok is False
    assert r2.detail["status"] == "failed"

    class _NoProbeBackend:
        backend_name = "transformers"
        model_id = "m2"

        @staticmethod
        def is_loaded():
            return True

    state.llm = _NoProbeBackend()
    r3 = await ldr.probe_model("m2")
    assert r3.ok is True
    assert r3.detail["is_loaded"] is True

    with pytest.raises(AppError):
        await ldr.probe_model("")


@pytest.mark.anyio
async def test_status_and_load_all_enabled_models(monkeypatch):
    ldr, state = _mk_loader()
    mm = MultiModelManager(
        models={"m1": _Backend(model_id="m1"), "m2": _Backend(model_id="m2")},
        default_id="m1",
    )
    state.llm = mm
    state.models_config = SimpleNamespace(primary_id="m1")

    # avoid re-locking deadlock path by monkeypatching load_model/load_default for this test
    async def _fake_load_model(mid, force=False):
        if mid == "m2":
            raise RuntimeError("bad model")
        return mod.LoadResult(model_id=mid, loaded=True, load_mode="lazy", detail={})

    monkeypatch.setattr(ldr, "load_model", _fake_load_model, raising=True)
    out = await ldr.load_all_enabled_models(force=False)
    assert out["m1"].loaded is True
    assert out["m2"].detail["status"] == "failed"

    st = await ldr.status()
    assert st["models_config_loaded"] is True
    assert st["registry_kind"] == "MultiModelManager"
