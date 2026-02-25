from __future__ import annotations

from dataclasses import dataclass

import pytest

from llm_server.core.errors import AppError
from llm_server.services.api_deps.core import auth

pytestmark = pytest.mark.unit


def test_rate_limit_exceeded(monkeypatch):
    auth.clear_rate_limit_state()
    monkeypatch.setattr(auth, "_role_rpm", lambda role: 1)
    monkeypatch.setattr(auth, "_now", lambda: 1000.0)

    auth._check_rate_limit("k1", None)

    with pytest.raises(AppError) as e:
        auth._check_rate_limit("k1", None)

    assert e.value.code == "rate_limited"
    assert e.value.status_code == 429
    assert "retry_after" in (e.value.extra or {})


def test_rate_limit_resets_after_window(monkeypatch):
    auth.clear_rate_limit_state()
    monkeypatch.setattr(auth, "_role_rpm", lambda role: 1)

    monkeypatch.setattr(auth, "_now", lambda: 1000.0)
    auth._check_rate_limit("k1", None)
    with pytest.raises(AppError):
        auth._check_rate_limit("k1", None)

    monkeypatch.setattr(auth, "_now", lambda: 1061.0)
    auth._check_rate_limit("k1", None)


@dataclass
class _Key:
    key: str = "x"
    quota_monthly: int | None = 2
    quota_used: int | None = 0


def test_quota_consumption_and_exhaustion():
    k = _Key(quota_monthly=2, quota_used=0)

    auth._check_and_consume_quota_in_session(k)
    assert k.quota_used == 1

    auth._check_and_consume_quota_in_session(k)
    assert k.quota_used == 2

    with pytest.raises(AppError) as e:
        auth._check_and_consume_quota_in_session(k)

    assert e.value.code == "quota_exhausted"
    assert e.value.status_code == 402
