from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from llm_server.telemetry import queries as q


class _Scalars:
    def __init__(self, values):
        self._values = values

    def all(self):
        return list(self._values)


class _Result:
    def __init__(self, *, one=None, all_rows=None, scalar_one=None, scalar_one_or_none=None, scalars=None):
        self._one = one
        self._all = all_rows or []
        self._scalar_one = scalar_one
        self._scalar_one_or_none = scalar_one_or_none
        self._scalars = scalars or []

    def one(self):
        return self._one

    def all(self):
        return list(self._all)

    def scalar_one(self):
        return self._scalar_one

    def scalar_one_or_none(self):
        return self._scalar_one_or_none

    def scalars(self):
        return _Scalars(self._scalars)


class _Session:
    def __init__(self, *, execute_results=None, scalar_results=None, role_rows=None):
        self._execute_results = list(execute_results or [])
        self._scalar_results = list(scalar_results or [])
        self._role_rows = role_rows or {}

    async def execute(self, stmt):
        if not self._execute_results:
            raise RuntimeError("no execute result queued")
        return self._execute_results.pop(0)

    async def scalar(self, stmt):
        if not self._scalar_results:
            raise RuntimeError("no scalar result queued")
        return self._scalar_results.pop(0)

    async def get(self, model, role_id):
        return self._role_rows.get(role_id)


def test_small_helpers():
    assert q._pct([], 50) is None
    assert q._pct([1.0, 2.0, 3.0], 0) == 1.0
    assert q._pct([1.0, 2.0, 3.0], 100) == 3.0
    assert q._safe_int("3") == 3
    assert q._safe_int("x") == 0
    assert q._safe_float("2.5") == 2.5
    assert q._safe_float("x") is None
    assert q._utc_iso_z().endswith("Z")


@pytest.mark.anyio
async def test_fetch_role_name_and_me_usage():
    session = _Session(
        execute_results=[
            _Result(one=(5, None, None, 12, 7)),
        ],
        role_rows={1: SimpleNamespace(name="admin")},
    )
    assert await q.fetch_role_name(session, role_id=1) == "admin"
    assert await q.fetch_role_name(session, role_id=None) is None

    usage = await q.get_me_usage(session, api_key_value="k1", role_name="admin")
    assert usage.api_key == "k1"
    assert usage.total_requests == 5
    assert usage.total_prompt_tokens == 12


@pytest.mark.anyio
async def test_get_admin_usage_and_list_api_keys():
    k1 = SimpleNamespace(key="key-abc-123", name="svc-a", role=SimpleNamespace(name="admin"), created_at=datetime.now(timezone.utc), disabled_at=None)
    k2 = SimpleNamespace(key="key-def-456", name="svc-b", role=None, created_at=datetime.now(timezone.utc), disabled_at=datetime.now(timezone.utc))
    session = _Session(
        execute_results=[
            _Result(all_rows=[("key-abc-123", 3, None, None, 10, 20)]),
            _Result(scalars=[k1]),
            _Result(scalar_one=2),
            _Result(scalars=[k1, k2]),
        ],
    )
    rows = await q.get_admin_usage(session)
    assert len(rows) == 1
    assert rows[0].role == "admin"

    page = await q.list_api_keys(session, limit=50, offset=0)
    assert page.total == 2
    assert page.items[0].key_prefix == "key-abc-"
    assert page.items[1].disabled is True


@pytest.mark.anyio
async def test_list_logs_and_admin_stats_and_reload_key():
    now = datetime.now(timezone.utc)
    log_row = SimpleNamespace(id=1, created_at=now)
    key_obj = SimpleNamespace(id=9, key="key-1", role=SimpleNamespace(name="admin"))
    session = _Session(
        execute_results=[
            _Result(scalars=[log_row]),
            _Result(one=(8, 100, 50, 12.0)),
            _Result(all_rows=[("m1", 8, 100, 50, 12.0)]),
            _Result(scalar_one_or_none=key_obj),
        ],
        scalar_results=[1],
    )

    logs = await q.list_inference_logs(
        session,
        model_id="m1",
        api_key_value="k1",
        route="/v1/generate",
        from_ts=None,
        to_ts=None,
        limit=10,
        offset=0,
        status_code_min=400,
        status_code_max=599,
        error_code="e",
        error_stage="s",
        cached=True,
    )
    assert logs.total == 1
    assert len(logs.items) == 1

    stats = await q.get_admin_stats(session, window_days=7)
    assert stats.total_requests == 8
    assert stats.per_model[0].model_id == "m1"

    reloaded = await q.reload_key_with_role(session, api_key_id=9)
    assert reloaded is key_obj


@pytest.mark.anyio
async def test_compute_window_generate_slo_contracts_payload():
    row = SimpleNamespace(
        n=4,
        errors=1,
        avg_latency_ms=100.0,
        max_latency_ms=230.0,
        prompt_avg=12.5,
        prompt_max=30,
        completion_avg=20.0,
        completion_max=44,
    )
    session = _Session(
        execute_results=[
            _Result(one=row),
            _Result(all_rows=[(10.0,), (20.0,), (30.0,)]),
            _Result(all_rows=[(11.0,), (22.0,), (44.0,)]),
        ]
    )
    payload = await q.compute_window_generate_slo_contracts_payload(
        session,
        window_seconds=300,
        routes=["/v1/generate"],
        model_id="m1",
    )
    assert payload["schema_version"] == q.RUNTIME_GENERATE_SLO_VERSION
    assert payload["window_seconds"] == 300
    assert payload["routes"] == ["/v1/generate"]
    assert payload["totals"]["requests"]["total"] == 4
    assert payload["totals"]["errors"]["total"] == 1
    assert payload["totals"]["latency_ms"]["p95"] == 30.0
    assert payload["models"][0]["model_id"] == "m1"
