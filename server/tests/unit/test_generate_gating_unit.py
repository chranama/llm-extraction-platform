from __future__ import annotations

import asyncio

import pytest

from llm_server.core.errors import AppError
from llm_server.services.limits.config import GenerateGateConfig
from llm_server.services.limits import generate_gating as gg


class _Metric:
    def labels(self, **kwargs):
        return self

    def inc(self, *_args, **_kwargs):
        return None

    def dec(self, *_args, **_kwargs):
        return None

    def set(self, *_args, **_kwargs):
        return None

    def observe(self, *_args, **_kwargs):
        return None


@pytest.fixture(autouse=True)
def _patch_metrics(monkeypatch):
    m = _Metric()
    for name in (
        "GENERATE_EXECUTION_SECONDS",
        "GENERATE_GATE_ENTERS",
        "GENERATE_GATE_REJECTS",
        "GENERATE_GATE_STARTS",
        "GENERATE_GATE_TIMEOUTS",
        "GENERATE_IN_FLIGHT",
        "GENERATE_QUEUE_DEPTH",
        "GENERATE_QUEUE_WAIT_SECONDS",
    ):
        monkeypatch.setattr(gg, name, m, raising=True)


@pytest.mark.anyio
async def test_token_helpers():
    sem = asyncio.Semaphore(1)
    tok = gg._SemaphoreToken(sem)
    assert tok.try_acquire() is True
    tok.release_if_held_sync()
    assert await tok.acquire_with_timeout(0.01) is True
    await tok.release_if_held()

    n = gg._NullToken()
    assert n.try_acquire() is True
    assert await n.acquire_with_timeout(0.0) is True
    await n.release_if_held()


@pytest.mark.anyio
async def test_gate_disabled_bypasses():
    gate = gg.GenerateGate(GenerateGateConfig(enabled=False, max_concurrent=1, max_queue=1, timeout_seconds=1.0, fail_fast=True, count_queued_as_in_flight=False))

    async def _fn():
        return "ok"

    assert await gate.run(_fn) == "ok"


@pytest.mark.anyio
async def test_gate_fail_fast_queue_full_and_concurrency_full():
    gate_q = gg.GenerateGate(GenerateGateConfig(enabled=True, max_concurrent=1, max_queue=1, timeout_seconds=1.0, fail_fast=True, count_queued_as_in_flight=False))
    gate_q._state.queue_slots._value = 0  # type: ignore[attr-defined]

    async def _fn():
        return "x"

    with pytest.raises(AppError) as e1:
        await gate_q.run(_fn)
    assert e1.value.code == "generate_overloaded"
    assert e1.value.status_code == 429
    assert e1.value.extra["max_queue"] == 1

    gate_c = gg.GenerateGate(GenerateGateConfig(enabled=True, max_concurrent=1, max_queue=0, timeout_seconds=1.0, fail_fast=True, count_queued_as_in_flight=False))
    gate_c._state.sem._value = 0  # type: ignore[attr-defined]
    with pytest.raises(AppError) as e2:
        await gate_c.run(_fn)
    assert e2.value.code == "generate_overloaded"
    assert e2.value.extra["max_concurrent"] == 1


@pytest.mark.anyio
async def test_gate_timeouts_and_success():
    async def _slow():
        await asyncio.sleep(0.05)
        return "late"

    gate_exec = gg.GenerateGate(GenerateGateConfig(enabled=True, max_concurrent=1, max_queue=1, timeout_seconds=0.01, fail_fast=False, count_queued_as_in_flight=False))
    with pytest.raises(AppError) as e1:
        await gate_exec.run(_slow)
    assert e1.value.code == "generate_overloaded"
    assert e1.value.extra["stage"] in ("execution", "queue_wait")

    gate_qwait = gg.GenerateGate(GenerateGateConfig(enabled=True, max_concurrent=1, max_queue=1, timeout_seconds=0.01, fail_fast=False, count_queued_as_in_flight=False))
    gate_qwait._state.sem._value = 0  # type: ignore[attr-defined]
    with pytest.raises(AppError) as e2:
        await gate_qwait.run(_slow)
    assert e2.value.code == "generate_overloaded"
    assert e2.value.extra["stage"] == "queue_wait"

    gate_ok = gg.GenerateGate(GenerateGateConfig(enabled=True, max_concurrent=2, max_queue=2, timeout_seconds=1.0, fail_fast=False, count_queued_as_in_flight=True))

    async def _fast():
        return 42

    assert await gate_ok.run(_fast) == 42
    snap = gate_ok.snapshot()
    assert snap.enabled is True
    assert snap.max_concurrent == 2
    assert snap.max_queue == 2


def test_singleton_get_and_reset(monkeypatch):
    gg.reset_generate_gate_for_tests()
    monkeypatch.setattr(
        gg,
        "load_generate_gate_config",
        lambda settings=None: GenerateGateConfig(enabled=True, max_concurrent=1, max_queue=1, timeout_seconds=1.0, fail_fast=True, count_queued_as_in_flight=False),
        raising=True,
    )
    g1 = gg.get_generate_gate()
    g2 = gg.get_generate_gate()
    assert g1 is g2
    gg.reset_generate_gate_for_tests()
    g3 = gg.get_generate_gate()
    assert g3 is not g1
