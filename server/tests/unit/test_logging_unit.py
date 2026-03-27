from __future__ import annotations

import importlib
import json
import logging
import time

import pytest
from starlette.requests import Request
from starlette.responses import Response


def _request(path="/v1/extract", headers=None):
    hdrs = []
    for k, v in (headers or {}).items():
        hdrs.append((k.lower().encode(), str(v).encode()))
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "path": path,
        "raw_path": path.encode(),
        "headers": hdrs,
        "query_string": b"",
        "client": ("127.0.0.1", 1234),
        "server": ("testserver", 80),
        "scheme": "http",
    }
    return Request(scope)


def test_json_formatter_includes_trace_and_job_fields():
    mod = importlib.import_module("llm_server.core.logging")
    formatter = mod.JsonFormatter()

    record = logging.LogRecord(
        name="llm_server.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="event",
        args=(),
        exc_info=None,
    )
    record.request_id = "req-1"
    record.trace_id = "trace-1"
    record.job_id = "job-1"

    payload = json.loads(formatter.format(record))
    assert payload["request_id"] == "req-1"
    assert payload["trace_id"] == "trace-1"
    assert payload["job_id"] == "job-1"


@pytest.mark.anyio
async def test_request_logging_middleware_includes_trace_id(monkeypatch):
    mod = importlib.reload(importlib.import_module("llm_server.core.logging"))

    captured = {}

    def _capture(message, *, extra=None):
        captured["message"] = message
        captured["extra"] = extra or {}

    monkeypatch.setattr(mod.access_logger, "info", _capture)

    async def _app(scope, receive, send):
        return None

    mw = mod.RequestLoggingMiddleware(_app)
    req = _request()
    req.state.start_ts = time.time()
    req.state.request_id = "req-123"
    req.state.trace_id = "trace-123"
    req.state.route = "/v1/extract"
    req.state.model_id = "fake-model"
    req.state.cached = False

    async def _call_next(request):
        return Response("ok", status_code=200)

    await mw.dispatch(req, _call_next)

    assert captured["message"] == "request"
    assert captured["extra"]["request_id"] == "req-123"
    assert captured["extra"]["trace_id"] == "trace-123"
    assert captured["extra"]["route"] == "/v1/extract"
