from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_rate_limit_exceeded(monkeypatch):
    import llm_server.api.deps as deps

    deps._RL.clear()
    monkeypatch.setattr(deps, "_role_rpm", lambda role: 1, raising=True)
    monkeypatch.setattr(deps, "_now", lambda: 1000.0, raising=True)

    # first ok
    deps._check_rate_limit("k1", None)

    # second in same window -> 429
    with pytest.raises(deps.AppError) as e:
        deps._check_rate_limit("k1", None)
    assert e.value.code == "rate_limited"
    assert e.value.status_code == 429
    assert "retry_after" in (e.value.extra or {})


def test_rate_limit_resets_after_window(monkeypatch):
    import llm_server.api.deps as deps

    deps._RL.clear()
    monkeypatch.setattr(deps, "_role_rpm", lambda role: 1, raising=True)

    # first window
    monkeypatch.setattr(deps, "_now", lambda: 1000.0, raising=True)
    deps._check_rate_limit("k1", None)
    with pytest.raises(deps.AppError):
        deps._check_rate_limit("k1", None)

    # after 60s -> new window should allow
    monkeypatch.setattr(deps, "_now", lambda: 1061.0, raising=True)
    deps._check_rate_limit("k1", None)


def test_quota_consumption_and_exhaustion():
    import llm_server.api.deps as deps
    from llm_server.db.models import ApiKey

    k = ApiKey(key="x", active=True, quota_monthly=2, quota_used=0)

    deps._check_and_consume_quota_in_session(k)
    assert k.quota_used == 1

    deps._check_and_consume_quota_in_session(k)
    assert k.quota_used == 2

    with pytest.raises(deps.AppError) as e:
        deps._check_and_consume_quota_in_session(k)
    assert e.value.code == "quota_exhausted"
    assert e.value.status_code == 402