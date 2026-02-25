from __future__ import annotations

from types import SimpleNamespace

import pytest

from llm_server.services.api_deps.health import readiness as rd


def _req():
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))


@pytest.mark.anyio
async def test_model_ready_check_skips_when_load_mode_off(monkeypatch):
    monkeypatch.setattr(rd, "effective_model_load_mode_from_request", lambda request: "off", raising=True)
    ok, status, details = await rd.model_ready_check_async(_req())
    assert (ok, status) == (True, "skipped (model_load_mode=off)")
    assert details["mode"] == "off"


@pytest.mark.anyio
async def test_model_ready_check_skips_when_readiness_off(monkeypatch):
    monkeypatch.setattr(rd, "effective_model_load_mode_from_request", lambda request: "lazy", raising=True)
    monkeypatch.setattr(
        rd, "resolve_default_model_id_and_backend_obj", lambda request: ("m1", "transformers", object()), raising=True
    )
    monkeypatch.setattr(rd, "per_model_readiness_mode", lambda request, model_id: "off", raising=True)

    ok, status, details = await rd.model_ready_check_async(_req())
    assert (ok, status) == (True, "skipped (readiness_mode=off)")
    assert details["model_id"] == "m1"


@pytest.mark.anyio
async def test_model_ready_check_external_backend_missing(monkeypatch):
    monkeypatch.setattr(rd, "effective_model_load_mode_from_request", lambda request: "lazy", raising=True)
    monkeypatch.setattr(rd, "resolve_default_model_id_and_backend_obj", lambda request: ("m1", "remote", None), raising=True)
    monkeypatch.setattr(rd, "per_model_readiness_mode", lambda request, model_id: "generate", raising=True)

    ok, status, details = await rd.model_ready_check_async(_req())
    assert (ok, status) == (False, "backend missing")
    assert details["backend"] == "remote"


@pytest.mark.anyio
async def test_model_ready_check_external_probe_paths(monkeypatch):
    monkeypatch.setattr(rd, "effective_model_load_mode_from_request", lambda request: "lazy", raising=True)
    monkeypatch.setattr(rd, "per_model_readiness_mode", lambda request, model_id: "probe", raising=True)

    backend = object()
    monkeypatch.setattr(
        rd, "resolve_default_model_id_and_backend_obj", lambda request: ("m1", "llamacpp", backend), raising=True
    )
    async def _llama_probe(_backend):
        return True, "ok", {"probe": "llama"}

    monkeypatch.setattr(rd, "llamacpp_dependency_check_async", _llama_probe, raising=True)
    ok1, st1, d1 = await rd.model_ready_check_async(_req())
    assert (ok1, st1) == (True, "ok")
    assert d1["probe"] == "llama"

    monkeypatch.setattr(rd, "resolve_default_model_id_and_backend_obj", lambda request: ("m2", "remote", backend), raising=True)
    async def _remote_probe(_backend):
        return False, "not ready", {"probe": "remote"}

    monkeypatch.setattr(rd, "remote_probe_async", _remote_probe, raising=True)
    ok2, st2, d2 = await rd.model_ready_check_async(_req())
    assert (ok2, st2) == (False, "not ready")
    assert d2["probe"] == "remote"


@pytest.mark.anyio
async def test_model_ready_check_external_generate_path(monkeypatch):
    monkeypatch.setattr(rd, "effective_model_load_mode_from_request", lambda request: "lazy", raising=True)
    monkeypatch.setattr(
        rd, "resolve_default_model_id_and_backend_obj", lambda request: ("m3", "remote", object()), raising=True
    )
    monkeypatch.setattr(rd, "per_model_readiness_mode", lambda request, model_id: "generate", raising=True)
    async def _generate_probe(_backend_obj):
        return True, "ok", {"checked": "generate"}

    monkeypatch.setattr(rd, "external_backend_generate_check_async", _generate_probe, raising=True)

    ok, status, details = await rd.model_ready_check_async(_req())
    assert (ok, status) == (True, "ok")
    assert details["checked"] == "generate"


@pytest.mark.anyio
async def test_model_ready_check_inproc_reflects_model_flags(monkeypatch):
    monkeypatch.setattr(rd, "effective_model_load_mode_from_request", lambda request: "lazy", raising=True)
    monkeypatch.setattr(
        rd, "resolve_default_model_id_and_backend_obj", lambda request: ("local-a", "transformers", object()), raising=True
    )
    monkeypatch.setattr(rd, "per_model_readiness_mode", lambda request, model_id: "generate", raising=True)
    monkeypatch.setattr(
        rd,
        "model_flags_from_app_state",
        lambda request: (True, None, "local-a", "local-a"),
        raising=True,
    )

    ok1, st1, d1 = await rd.model_ready_check_async(_req())
    assert (ok1, st1) == (True, "ok")
    assert d1["model_loaded"] is True

    monkeypatch.setattr(
        rd,
        "model_flags_from_app_state",
        lambda request: (True, "load error", "local-a", "local-a"),
        raising=True,
    )
    ok2, st2, d2 = await rd.model_ready_check_async(_req())
    assert (ok2, st2) == (False, "not ready")
    assert d2["model_error"] == "load error"
