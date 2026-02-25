from __future__ import annotations

from types import SimpleNamespace

import pytest

from llm_server.core import redis as rmod


class _Metric:
    def labels(self, **kwargs):
        return self

    def inc(self):
        return None

    def observe(self, value):
        return None


class _FakeRedis:
    def __init__(self):
        self.data = {}
        self.closed = False

    async def get(self, key):
        return self.data.get(key)

    async def set(self, key, value, ex=None):
        self.data[key] = value
        self.last_ex = ex

    async def aclose(self):
        self.closed = True


@pytest.fixture(autouse=True)
def _patch_metrics(monkeypatch):
    m = _Metric()
    monkeypatch.setattr(rmod, "LLM_REDIS_HITS", m, raising=True)
    monkeypatch.setattr(rmod, "LLM_REDIS_MISSES", m, raising=True)
    monkeypatch.setattr(rmod, "LLM_REDIS_LATENCY", m, raising=True)


@pytest.mark.anyio
async def test_init_redis_disabled_and_enabled(monkeypatch):
    monkeypatch.setattr(rmod, "get_settings", lambda: SimpleNamespace(redis_enabled=False, redis_url="redis://x"), raising=True)
    assert await rmod.init_redis() is None

    client = _FakeRedis()
    monkeypatch.setattr(rmod, "get_settings", lambda: SimpleNamespace(redis_enabled=True, redis_url="redis://x"), raising=True)
    monkeypatch.setattr(rmod, "from_url", lambda url, decode_responses=True: client, raising=True)
    out = await rmod.init_redis()
    assert out is client


@pytest.mark.anyio
async def test_close_redis_and_get_from_request():
    c = _FakeRedis()
    await rmod.close_redis(c)
    assert c.closed is True
    await rmod.close_redis(None)

    req = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(redis=c)))
    assert rmod.get_redis_from_request(req) is c


@pytest.mark.anyio
async def test_redis_get_and_set_paths():
    c = _FakeRedis()

    miss = await rmod.redis_get(c, "k1", model_id="m1", kind="single")
    assert miss is None

    await rmod.redis_set(c, "k1", "v1")
    hit = await rmod.redis_get(c, "k1", model_id="m1", kind="single")
    assert hit == "v1"

    await rmod.redis_set(c, "k2", "v2", ex=60)
    assert c.data["k2"] == "v2"
    assert c.last_ex == 60

    assert await rmod.redis_get(None, "k3") is None
    await rmod.redis_set(None, "k3", "v3")
