from __future__ import annotations

from types import SimpleNamespace

import pytest

from llm_server.telemetry import traces


class _Session:
    def __init__(self):
        self.rows = []
        self.commits = 0
        self.flushes = 0
        self.refreshes = 0

    def add(self, row):
        self.rows.append(row)

    async def commit(self):
        self.commits += 1

    async def flush(self):
        self.flushes += 1

    async def refresh(self, row):
        self.refreshes += 1


def test_compact_details_drops_complex_noise():
    data = traces.compact_details(
        {
            "schema_id": "ticket",
            "cache": True,
            "nested": {"layer": "redis", "bad": object()},
            "items": ["a", 1, object()],
            "raw": object(),
        }
    )
    assert data == {
        "schema_id": "ticket",
        "cache": True,
        "nested": {"layer": "redis"},
        "items": ["a", 1],
    }


@pytest.mark.anyio
async def test_record_trace_event_persists_compact_row():
    session = _Session()
    row = await traces.record_trace_event(
        session,
        trace_id="rid-1",
        event_name="extract.accepted",
        route="/v1/extract",
        stage="start",
        status="accepted",
        request_id="rid-1",
        job_id=None,
        model_id="fake",
        details={"schema_id": "ticket", "raw": object()},
        commit=True,
    )
    assert row.trace_id == "rid-1"
    assert row.event_name == "extract.accepted"
    assert row.details_json == {"schema_id": "ticket"}
    assert session.commits == 1
    assert session.refreshes == 1


def test_trace_id_from_ctx_prefers_trace_id():
    ctx = SimpleNamespace(state=SimpleNamespace(trace_id="trace-1", request_id="rid-1"))
    assert traces.trace_id_from_ctx(ctx) == "trace-1"
