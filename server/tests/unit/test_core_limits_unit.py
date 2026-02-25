from __future__ import annotations

from types import SimpleNamespace

import pytest
from starlette.requests import Request
from starlette.responses import Response

from llm_server.core import limits as lim
from llm_server.core.errors import AppError


def _req(path: str = "/v1/generate", method: str = "POST") -> Request:
    app = SimpleNamespace(state=SimpleNamespace())
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": method,
        "path": path,
        "raw_path": path.encode(),
        "headers": [],
        "query_string": b"",
        "app": app,
        "client": ("127.0.0.1", 1234),
        "server": ("testserver", 80),
        "scheme": "http",
    }
    req = Request(scope)
    req.state.request_id = "rid-1"
    return req


class _Metric:
    def labels(self, **kwargs):
        return self

    def inc(self, *_args, **_kwargs):
        return None


def test_read_helpers_and_container_limit(monkeypatch):
    monkeypatch.setattr(lim, "get_settings", lambda: SimpleNamespace(max_concurrent_requests=200), raising=True)
    assert lim._read_max_concurrency() == 64
    monkeypatch.setattr(lim, "get_settings", lambda: SimpleNamespace(max_concurrent_requests="x"), raising=True)
    assert lim._read_max_concurrency() == 2

    s = SimpleNamespace(container_memory_bytes="1024")
    assert lim._read_container_limit_bytes(s) == 1024
    assert lim._read_container_limit_bytes(SimpleNamespace(container_memory_bytes="bad")) is None

    assert lim._read_mem_guard_enabled(SimpleNamespace(mem_guard_enabled=True)) is True
    assert lim._read_mem_guard_rss_pct(SimpleNamespace(mem_guard_rss_pct=0.01)) == 0.10
    assert lim._read_mem_guard_rss_pct(SimpleNamespace(mem_guard_rss_pct=2.0)) == 0.99


def test_maybe_trip_memory_guard(monkeypatch):
    m = _Metric()
    monkeypatch.setattr(lim, "LLM_GUARD_TRIPS", m, raising=True)
    req = _req()
    settings = SimpleNamespace(mem_guard_enabled=True, container_memory_bytes=1000, mem_guard_rss_pct=0.8)
    monkeypatch.setattr(lim, "_read_rss_bytes", lambda: 850, raising=True)

    with pytest.raises(AppError) as e:
        lim._maybe_trip_memory_guard(req, settings)
    assert e.value.code == "server_overloaded"
    assert e.value.status_code == 503

    settings2 = SimpleNamespace(mem_guard_enabled=False, container_memory_bytes=1000, mem_guard_rss_pct=0.8)
    lim._maybe_trip_memory_guard(req, settings2)


@pytest.mark.anyio
async def test_guard_middleware_dispatch_paths(monkeypatch):
    async def _app(scope, receive, send):
        return None

    monkeypatch.setattr(lim, "get_settings", lambda: SimpleNamespace(max_concurrent_requests=2, mem_guard_enabled=False, mem_guard_rss_pct=0.85, container_memory_bytes=None), raising=True)
    mw = lim._GuardMiddleware(_app)

    calls = {"n": 0}

    async def _call_next(request):
        calls["n"] += 1
        return Response("ok", status_code=200)

    # non-heavy path bypasses guard logic
    r1 = await mw.dispatch(_req(path="/healthz", method="GET"), _call_next)
    assert r1.status_code == 200

    # heavy path triggers memory check hook and call_next
    monkeypatch.setattr(lim, "_maybe_trip_memory_guard", lambda request, settings: None, raising=True)
    r2 = await mw.dispatch(_req(path="/v1/generate", method="POST"), _call_next)
    assert r2.status_code == 200
    assert calls["n"] >= 2


def test_setup_adds_middleware():
    added = {}

    class _App:
        @staticmethod
        def add_middleware(cls):
            added["cls"] = cls

    lim.setup(_App())
    assert added["cls"] is lim._GuardMiddleware
