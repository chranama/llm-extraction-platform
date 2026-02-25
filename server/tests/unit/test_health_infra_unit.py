from __future__ import annotations

from types import SimpleNamespace

import pytest

from llm_server.services.api_deps.health import infra


def _req():
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))


def test_settings_from_request_prefers_app_state(monkeypatch):
    req = _req()
    req.app.state.settings = SimpleNamespace(redis_enabled=True)
    assert infra.settings_from_request(req).redis_enabled is True

    monkeypatch.setattr(infra, "get_settings", lambda: SimpleNamespace(redis_enabled=False), raising=True)
    req2 = _req()
    assert infra.settings_from_request(req2).redis_enabled is False


@pytest.mark.anyio
async def test_db_check_paths():
    class _S1:
        async def execute(self, stmt):
            return 1

    ok, status = await infra.db_check(_S1())
    assert (ok, status) == (True, "ok")

    class _S2:
        async def execute(self, stmt):
            raise RuntimeError("db down")

    ok2, status2 = await infra.db_check(_S2())
    assert (ok2, status2) == (False, "error")


@pytest.mark.anyio
async def test_redis_check_paths(monkeypatch):
    req = _req()
    monkeypatch.setattr(infra, "settings_from_request", lambda request: SimpleNamespace(redis_enabled=False), raising=True)
    ok0, st0 = await infra.redis_check(req)
    assert (ok0, st0) == (True, "disabled")

    monkeypatch.setattr(infra, "settings_from_request", lambda request: SimpleNamespace(redis_enabled=True), raising=True)
    monkeypatch.setattr(infra, "get_redis_from_request", lambda request: None, raising=True)
    ok1, st1 = await infra.redis_check(req)
    assert (ok1, st1) == (False, "not initialized")

    class _Rok:
        async def ping(self):
            return True

    monkeypatch.setattr(infra, "get_redis_from_request", lambda request: _Rok(), raising=True)
    ok2, st2 = await infra.redis_check(req)
    assert (ok2, st2) == (True, "ok")

    class _Rweird:
        async def ping(self):
            return "PONG"

    monkeypatch.setattr(infra, "get_redis_from_request", lambda request: _Rweird(), raising=True)
    ok3, st3 = await infra.redis_check(req)
    assert ok3 is False
    assert st3.startswith("unexpected:")

    class _Rbad:
        async def ping(self):
            raise RuntimeError("boom")

    monkeypatch.setattr(infra, "get_redis_from_request", lambda request: _Rbad(), raising=True)
    ok4, st4 = await infra.redis_check(req)
    assert (ok4, st4) == (False, "error")
