from __future__ import annotations

from types import SimpleNamespace

import pytest
from starlette.requests import Request
from starlette.responses import Response

from llm_server.core.errors import AppError
from llm_server.services.limits import early_reject_middleware as erm


def _mk_request(app, path="/v1/generate"):
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "path": path,
        "raw_path": path.encode(),
        "headers": [],
        "query_string": b"",
        "app": app,
        "client": ("127.0.0.1", 1234),
        "server": ("testserver", 80),
        "scheme": "http",
    }
    return Request(scope)


def test_cfg_from_app_and_cache_invalidation():
    state = SimpleNamespace(settings=SimpleNamespace(limits={"generate_early_reject": {"enabled": True, "reject_queue_depth_gte": 1, "reject_in_flight_gte": 2, "routes": ["/v1/generate"]}}))
    app = SimpleNamespace(state=state)

    c1 = erm._cfg_from_app(app)
    c2 = erm._cfg_from_app(app)
    assert c1 is c2
    assert c1.reject_in_flight_gte == 2

    # change settings object => fingerprint changes => recompute
    app.state.settings = SimpleNamespace(limits={"generate_early_reject": {"enabled": False}})
    c3 = erm._cfg_from_app(app)
    assert c3.enabled is False

    # missing settings falls back to defaults
    app2 = SimpleNamespace(state=SimpleNamespace(settings=None))
    c4 = erm._cfg_from_app(app2)
    assert c4.enabled is True


@pytest.mark.anyio
async def test_middleware_dispatch_paths(monkeypatch):
    async def _app(scope, receive, send):
        return None

    mw = erm.EarlyRejectGenerateMiddleware(_app)
    app = SimpleNamespace(state=SimpleNamespace(settings=SimpleNamespace(limits={"generate_early_reject": {"enabled": True, "reject_queue_depth_gte": 3, "reject_in_flight_gte": 2, "routes": ["/v1/generate"]}})))

    async def _call_next(request):
        return Response("ok", status_code=200)

    # non-matching route bypass
    req1 = _mk_request(app, path="/healthz")
    r1 = await mw.dispatch(req1, _call_next)
    assert r1.status_code == 200

    # gate disabled => reject
    monkeypatch.setattr(
        erm,
        "get_generate_gate",
        lambda: SimpleNamespace(snapshot=lambda: SimpleNamespace(enabled=False, in_flight_estimate=0, queue_depth_estimate=0, max_concurrent=1, max_queue=1)),
        raising=True,
    )
    req2 = _mk_request(app, path="/v1/generate")
    with pytest.raises(AppError) as e2:
        await mw.dispatch(req2, _call_next)
    assert e2.value.code == "generate_overloaded"
    assert e2.value.extra["reason"] == "disabled"

    # in-flight threshold => reject
    monkeypatch.setattr(
        erm,
        "get_generate_gate",
        lambda: SimpleNamespace(snapshot=lambda: SimpleNamespace(enabled=True, in_flight_estimate=2, queue_depth_estimate=0, max_concurrent=1, max_queue=1)),
        raising=True,
    )
    with pytest.raises(AppError) as e3:
        await mw.dispatch(_mk_request(app, "/v1/generate"), _call_next)
    assert e3.value.extra["reason"] == "in_flight_high"

    # queue threshold => reject
    app.state.settings = SimpleNamespace(limits={"generate_early_reject": {"enabled": True, "reject_queue_depth_gte": 1, "reject_in_flight_gte": 0, "routes": ["/v1/generate"]}})
    monkeypatch.setattr(
        erm,
        "get_generate_gate",
        lambda: SimpleNamespace(snapshot=lambda: SimpleNamespace(enabled=True, in_flight_estimate=0, queue_depth_estimate=1, max_concurrent=1, max_queue=1)),
        raising=True,
    )
    with pytest.raises(AppError) as e4:
        await mw.dispatch(_mk_request(app, "/v1/generate"), _call_next)
    assert e4.value.extra["reason"] == "queue_full"

    # healthy path => pass through
    monkeypatch.setattr(
        erm,
        "get_generate_gate",
        lambda: SimpleNamespace(snapshot=lambda: SimpleNamespace(enabled=True, in_flight_estimate=0, queue_depth_estimate=0, max_concurrent=1, max_queue=1)),
        raising=True,
    )
    r5 = await mw.dispatch(_mk_request(app, "/v1/generate"), _call_next)
    assert r5.status_code == 200
