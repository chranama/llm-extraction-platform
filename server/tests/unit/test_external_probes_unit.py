from __future__ import annotations

import pytest

from llm_server.services.api_deps.health import external_probes as probes


class _Client:
    def __init__(self, health_value=None):
        self._health_value = health_value

    def health(self):
        return self._health_value


class _IsReadyBackend:
    def __init__(self, value):
        self._value = value

    def is_ready(self):
        if callable(self._value):
            return self._value()
        return self._value


class _CanGenerateBackend:
    def __init__(self, value):
        self._value = value

    def can_generate(self):
        if callable(self._value):
            return self._value()
        return self._value


class _ClientOnlyBackend:
    def __init__(self, client):
        self._client = client


def test_sync_llamacpp_dependency_check_prefers_is_ready():
    b = _IsReadyBackend((True, {"path": "is_ready"}))
    ok, status, details = probes.sync_llamacpp_dependency_check(b)
    assert ok is True
    assert status == "ok"
    assert details == {"path": "is_ready"}


def test_sync_llamacpp_dependency_check_health_fallback_and_missing():
    ok1, st1, d1 = probes.sync_llamacpp_dependency_check(_ClientOnlyBackend(_Client({"status": "ok"})))
    assert (ok1, st1) == (True, "ok")
    assert d1["health"]["status"] == "ok"

    ok2, st2, d2 = probes.sync_llamacpp_dependency_check(object())
    assert (ok2, st2) == (False, "missing health check")
    assert "reason" in d2


def test_sync_external_backend_generate_check_paths():
    b1 = _CanGenerateBackend((True, {"source": "can_generate"}))
    ok1, st1, d1 = probes.sync_external_backend_generate_check(b1)
    assert (ok1, st1) == (True, "ok")
    assert d1 == {"source": "can_generate"}

    b2 = _IsReadyBackend((False, {"source": "is_ready"}))
    ok2, st2, d2 = probes.sync_external_backend_generate_check(b2)
    assert (ok2, st2) == (False, "not ready")
    assert d2["source"] == "is_ready"
    assert "note" in d2

    ok3, st3, d3 = probes.sync_external_backend_generate_check(object())
    assert (ok3, st3) == (False, "missing readiness probe")
    assert "reason" in d3


def test_sync_remote_probe_paths_and_exception():
    ok1, st1, d1 = probes.sync_remote_probe(_IsReadyBackend((True, {"v": 1})))
    assert (ok1, st1, d1) == (True, "ok", {"v": 1})

    ok2, st2, d2 = probes.sync_remote_probe(_ClientOnlyBackend(_Client({"healthy": True})))
    assert (ok2, st2) == (True, "ok")
    assert d2["health"]["healthy"] is True

    ok3, st3, d3 = probes.sync_remote_probe(object())
    assert (ok3, st3) == (False, "missing remote probe")
    assert "reason" in d3

    def boom():
        raise RuntimeError("probe exploded")

    ok4, st4, d4 = probes.sync_remote_probe(_IsReadyBackend(boom))
    assert (ok4, st4) == (False, "error")
    assert "probe exploded" in d4["error"]


@pytest.mark.anyio
async def test_async_probe_wrappers_delegate_to_sync():
    backend = _IsReadyBackend((True, {"ok": 1}))

    ok1, st1, _ = await probes.llamacpp_dependency_check_async(backend)
    ok2, st2, _ = await probes.external_backend_generate_check_async(_CanGenerateBackend((False, {"ok": 2})))
    ok3, st3, _ = await probes.remote_probe_async(backend)

    assert (ok1, st1) == (True, "ok")
    assert (ok2, st2) == (False, "not ready")
    assert (ok3, st3) == (True, "ok")
