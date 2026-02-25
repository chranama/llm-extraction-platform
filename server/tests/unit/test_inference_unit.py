from __future__ import annotations

from types import SimpleNamespace

import pytest
from sqlalchemy.exc import IntegrityError

from llm_server.services.llm_runtime import inference as inf


class _Metric:
    def labels(self, **kwargs):
        return self

    def inc(self, *_args, **_kwargs):
        return None

    def observe(self, *_args, **_kwargs):
        return None


class _Session:
    def __init__(self):
        self.added = []
        self.did_flush = False
        self.did_rollback = False
        self.did_commit = False
        self.exec_result = None
        self.flush_exc = None

    def add(self, obj):
        self.added.append(obj)

    async def flush(self):
        self.did_flush = True
        if self.flush_exc:
            raise self.flush_exc

    async def rollback(self):
        self.did_rollback = True

    async def commit(self):
        self.did_commit = True

    async def execute(self, stmt):
        return self.exec_result


class _ScalarOneOrNone:
    def __init__(self, obj):
        self._obj = obj

    def scalar_one_or_none(self):
        return self._obj


@pytest.fixture(autouse=True)
def _patch_metrics(monkeypatch):
    m = _Metric()
    monkeypatch.setattr(inf, "LLM_REDIS_HITS", m, raising=True)
    monkeypatch.setattr(inf, "LLM_REDIS_MISSES", m, raising=True)
    monkeypatch.setattr(inf, "LLM_REDIS_LATENCY", m, raising=True)
    monkeypatch.setattr(inf, "LLM_TOKENS", m, raising=True)


def test_set_request_meta_and_parse_redis_output():
    req = SimpleNamespace(state=SimpleNamespace())
    inf.set_request_meta(req, route="/v1/generate", model_id="m1", cached=True)
    assert req.state.route == "/v1/generate"
    assert req.state.model_id == "m1"
    assert req.state.cached is True

    assert inf._parse_redis_cached_output(None) is None
    assert inf._parse_redis_cached_output('{"output":"x"}') == "x"
    assert inf._parse_redis_cached_output('{"output":""}') is None
    assert inf._parse_redis_cached_output("not-json") is None


@pytest.mark.anyio
async def test_cache_reads_and_writes(monkeypatch):
    cache = inf.CacheSpec(model_id="m1", prompt="p", prompt_hash="h", params_fp="fp", redis_key="rk", redis_ttl_seconds=10)
    session = _Session()

    # Redis hit
    async def _redis_get_hit(redis, key, *, model_id="unknown", kind="single"):
        return '{"output":"from-redis"}'

    monkeypatch.setattr(inf, "redis_get", _redis_get_hit, raising=True)
    out, cached, layer = await inf.get_cached_output(session, redis=object(), cache=cache, kind="single", enabled=True)
    assert (out, cached, layer) == ("from-redis", True, "redis")

    # DB hit + redis backfill
    session.exec_result = _ScalarOneOrNone(SimpleNamespace(output="from-db"))
    async def _redis_get_miss(redis, key, *, model_id="unknown", kind="single"):
        return None
    writes = []
    async def _redis_set(redis, key, value, *, ex=None):
        writes.append((key, value, ex))

    monkeypatch.setattr(inf, "redis_get", _redis_get_miss, raising=True)
    monkeypatch.setattr(inf, "redis_set", _redis_set, raising=True)
    out2, cached2, layer2 = await inf.get_cached_output(session, redis=object(), cache=cache, kind="single", enabled=True)
    assert (out2, cached2, layer2) == ("from-db", True, "db")
    assert writes and writes[0][2] == 10

    # Disabled
    out3, cached3, layer3 = await inf.get_cached_output(session, redis=None, cache=cache, kind="single", enabled=False)
    assert (out3, cached3, layer3) == (None, False, None)


@pytest.mark.anyio
async def test_write_cache_and_write_logs(monkeypatch):
    cache = inf.CacheSpec(model_id="m1", prompt="p", prompt_hash="h", params_fp="fp", redis_key="rk")
    session = _Session()

    writes = []
    async def _redis_set(redis, key, value, *, ex=None):
        writes.append((key, ex))

    monkeypatch.setattr(inf, "redis_set", _redis_set, raising=True)

    await inf.write_cache(session, redis=object(), cache=cache, output="out", enabled=True)
    assert session.did_flush is True
    assert writes and writes[0][0] == "rk"

    # IntegrityError path rolls back
    session2 = _Session()
    session2.flush_exc = IntegrityError("stmt", "params", "orig")
    await inf.write_cache(session2, redis=None, cache=cache, output="out", enabled=True)
    assert session2.did_rollback is True

    # write_inference_log clamps status and commits
    session3 = _Session()
    await inf.write_inference_log(
        session3,
        api_key="k",
        request_id="r1",
        route="/v1/generate",
        client_host="127.0.0.1",
        model_id="m1",
        params_json={"a": 1},
        prompt="p",
        output="o",
        latency_ms=12.0,
        prompt_tokens=3,
        completion_tokens=2,
        status_code=0,
        cached=True,
        error_code="  e1  ",
        error_stage="  parse  ",
        commit=True,
    )
    assert session3.did_commit is True
    row = session3.added[-1]
    assert row.status_code == 500
    assert row.error_code == "e1"
    assert row.error_stage == "parse"

    # failure wrapper delegates with empty prompt/output
    session4 = _Session()
    await inf.write_failure_log(
        session4,
        api_key="k",
        request_id=None,
        route="/x",
        client_host=None,
        model_id="m1",
        latency_ms=None,
        status_code=503,
        error_code="backend_down",
        error_stage="inference",
        cached=False,
        commit=False,
    )
    row2 = session4.added[-1]
    assert row2.prompt == ""
    assert row2.output is None
    assert row2.status_code == 503
