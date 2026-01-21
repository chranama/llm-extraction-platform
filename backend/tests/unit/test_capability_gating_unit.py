# tests/unit/test_capability_gating_unit.py
from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from llm_server.api.capabilities import router as capabilities_router
from llm_server.api.generate import router as generate_router
from llm_server.core.errors import AppError


# ----------------------------
# Test helpers / fakes
# ----------------------------

@dataclass
class _FakeSettings:
    enable_generate: bool = True
    enable_extract: bool = True
    env: str = "test"

    # generate.py's resolve_model commonly references this.
    # Keep it present even if we stub resolve_model below.
    all_model_ids: list[str] = field(default_factory=lambda: ["test-model"])


class _DummyLLM:
    """
    Minimal LLM object to satisfy generate endpoint without external IO.

    IMPORTANT: generate() is sync because the /v1/generate endpoint calls it sync.
    """

    def __init__(self, text: str = "ok", model_id: str = "test-model"):
        self._text = text
        self.model_id = model_id

    def generate(self, prompt: str, **kwargs) -> str:
        return self._text


class _DummyAsyncSession:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _DummySessionmaker:
    def __call__(self):
        return _DummyAsyncSession()


def _app_error_handler(_request, exc: AppError):
    payload: dict[str, Any] = {
        "code": getattr(exc, "code", "app_error"),
        "message": str(getattr(exc, "message", "")) or str(exc),
    }
    extra = getattr(exc, "extra", None)
    if extra is not None:
        payload["extra"] = extra
    return JSONResponse(status_code=getattr(exc, "status_code", 500), content=payload)


def _exception_handler(_request, exc: Exception):
    # Test-only: surface the actual traceback instead of "Internal Server Error"
    return JSONResponse(
        status_code=500,
        content={
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        },
    )


# ----------------------------
# App factory
# ----------------------------

def _make_client(
    *,
    enable_generate: bool,
    enable_extract: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> TestClient:
    """
    FastAPI app with real routers, but overrides settings/auth/llm/db/cache/logging
    so the tests stay unit-scoped (no DB/HF/network).
    """
    app = FastAPI()
    app.add_exception_handler(AppError, _app_error_handler)
    app.add_exception_handler(Exception, _exception_handler)

    # ---- Patch settings used by /v1/capabilities + generate gating
    import llm_server.core.config as core_config
    import llm_server.api.capabilities as caps_api
    import llm_server.api.generate as gen_api

    fake_settings = _FakeSettings(enable_generate=enable_generate, enable_extract=enable_extract)

    def _fake_get_settings():
        return fake_settings

    monkeypatch.setattr(core_config, "get_settings", _fake_get_settings)
    monkeypatch.setattr(caps_api, "get_settings", _fake_get_settings)
    monkeypatch.setattr(gen_api, "get_settings", _fake_get_settings)

    # ---- Make model resolution trivial in unit tests
    # This avoids coupling to production Settings + LLM internals.
    def _fake_resolve_model(llm: Any, model_override: str | None):
        # Return the dummy model id + the llm itself as the model handle.
        return ("test-model", llm)

    monkeypatch.setattr(gen_api, "resolve_model", _fake_resolve_model, raising=True)

    # ---- Patch DB sessionmaker used in generate.py
    import llm_server.db.session as db_session
    monkeypatch.setattr(db_session, "get_sessionmaker", lambda: _DummySessionmaker())

    # ---- Patch caching functions to return correct shapes
    async def _fake_get_cached_output(*args, **kwargs):
        # expected: (cached_out, cached_flag, layer)
        return (None, False, None)

    async def _fake_write_cache(*args, **kwargs) -> None:
        return None

    monkeypatch.setattr(gen_api, "get_cached_output", _fake_get_cached_output, raising=False)
    monkeypatch.setattr(gen_api, "write_cache", _fake_write_cache, raising=False)

    # ---- Patch inference logging (avoid DB writes)
    async def _fake_write_inference_log(*args, **kwargs) -> None:
        return None

    monkeypatch.setattr(gen_api, "write_inference_log", _fake_write_inference_log, raising=False)

    # ---- Provide dummy LLM + auth via dependency override
    import llm_server.api.deps as api_deps

    async def _fake_get_api_key():
        class _K:
            key = "test-key"
            api_key = "test-key"

        return _K()

    async def _fake_get_llm():
        return _DummyLLM("ok", model_id="test-model")

    # IMPORTANT: override both the canonical deps function and whatever
    # generate.py is actually depending on (often imported into module scope).
    app.dependency_overrides[api_deps.get_api_key] = _fake_get_api_key
    app.dependency_overrides[api_deps.get_llm] = _fake_get_llm
    if hasattr(gen_api, "get_llm"):
        app.dependency_overrides[gen_api.get_llm] = _fake_get_llm

    # ---- Include routers
    app.include_router(capabilities_router)
    app.include_router(generate_router)

    # Critical: do not re-raise app exceptions into pytest
    return TestClient(app, raise_server_exceptions=False)


def _assert_ok_response(r):
    if r.status_code != 200:
        # try to print JSON traceback if present
        try:
            body = r.json()
        except Exception:
            body = r.text
        raise AssertionError(f"Expected 200, got {r.status_code}. Body: {body}")


# ----------------------------
# Tests
# ----------------------------

def test_capabilities_both_enabled(monkeypatch: pytest.MonkeyPatch):
    c = _make_client(enable_generate=True, enable_extract=True, monkeypatch=monkeypatch)
    r = c.get("/v1/capabilities")
    _assert_ok_response(r)
    body = r.json()
    assert body["generate"] is True
    assert body["extract"] is True
    assert body["mode"] == "generate+extract"


def test_capabilities_generate_only(monkeypatch: pytest.MonkeyPatch):
    c = _make_client(enable_generate=True, enable_extract=False, monkeypatch=monkeypatch)
    r = c.get("/v1/capabilities")
    _assert_ok_response(r)
    body = r.json()
    assert body["generate"] is True
    assert body["extract"] is False
    assert body["mode"] == "generate-only"


def test_capabilities_extract_only_reports_disabled_generate(monkeypatch: pytest.MonkeyPatch):
    c = _make_client(enable_generate=False, enable_extract=True, monkeypatch=monkeypatch)
    r = c.get("/v1/capabilities")
    _assert_ok_response(r)
    body = r.json()
    assert body["generate"] is False
    assert body["extract"] is True
    assert "mode" in body


def test_capabilities_endpoint_always_available(monkeypatch: pytest.MonkeyPatch):
    c = _make_client(enable_generate=False, enable_extract=False, monkeypatch=monkeypatch)
    r = c.get("/v1/capabilities")
    _assert_ok_response(r)


def test_generate_is_blocked_when_disabled(monkeypatch: pytest.MonkeyPatch):
    c = _make_client(enable_generate=False, enable_extract=False, monkeypatch=monkeypatch)
    r = c.post("/v1/generate", json={"prompt": "Say ok", "cache": False})

    assert r.status_code == 501, f"Expected 501, got {r.status_code}. Body: {r.json() if r.headers.get('content-type','').startswith('application/json') else r.text}"
    body = r.json()
    assert body.get("code") == "capability_disabled"
    assert body.get("extra", {}).get("capability") == "generate"


def test_generate_works_when_enabled(monkeypatch: pytest.MonkeyPatch):
    c = _make_client(enable_generate=True, enable_extract=False, monkeypatch=monkeypatch)
    r = c.post("/v1/generate", json={"prompt": "Say ok", "cache": False})

    _assert_ok_response(r)
    body = r.json()

    # tolerant to response schema
    for k in ("output", "text", "response", "completion"):
        if k in body:
            assert str(body[k]).strip().lower() == "ok"
            return
    assert "ok" in str(body).lower()


def test_generate_does_not_trigger_external_io_when_enabled(monkeypatch: pytest.MonkeyPatch):
    c = _make_client(enable_generate=True, enable_extract=True, monkeypatch=monkeypatch)
    r = c.post("/v1/generate", json={"prompt": "Say ok", "cache": False})
    _assert_ok_response(r)