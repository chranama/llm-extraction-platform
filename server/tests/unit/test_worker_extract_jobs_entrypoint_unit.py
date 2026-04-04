from __future__ import annotations

from types import SimpleNamespace

import pytest

import llm_server.worker.extract_jobs as worker_jobs


class _LifespanContext:
    def __init__(self, app):
        self._app = app

    async def __aenter__(self):
        return self._app

    async def __aexit__(self, exc_type, exc, tb):
        return False


@pytest.mark.anyio
async def test_run_worker_delegates_to_application_process_once(monkeypatch: pytest.MonkeyPatch):
    calls: list[dict[str, object]] = []
    app = SimpleNamespace(
        state=SimpleNamespace(db_sessionmaker="sessionmaker"),
        router=SimpleNamespace(lifespan_context=lambda _app: _LifespanContext(app)),
    )

    async def fake_process_extract_job_once(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(status="succeeded")

    monkeypatch.setattr(worker_jobs, "create_app", lambda: app, raising=True)
    monkeypatch.setattr(worker_jobs, "queue_from_app", lambda _app: "queue", raising=True)
    monkeypatch.setattr(
        worker_jobs,
        "process_extract_job_once",
        fake_process_extract_job_once,
        raising=True,
    )

    await worker_jobs.run_worker(once=True, poll_timeout_seconds=7)

    assert calls == [
        {
            "app": app,
            "sessionmaker": "sessionmaker",
            "queue": "queue",
            "timeout_seconds": 7,
        }
    ]
