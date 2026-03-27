from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

from llm_server.observability.regression_manifests import (
    REGRESSION_REPLAY_MANIFEST_VERSION,
    build_regression_replay_manifest,
)
from llm_server.observability.replay_cases import (
    build_replay_case_from_inference_log,
    build_replay_case_from_trace,
)
from llm_server.telemetry.types import TraceDetail, TraceEventView


def test_build_replay_case_from_inference_log_extract_success():
    log = SimpleNamespace(
        id=7,
        created_at=datetime(2026, 3, 27, tzinfo=UTC),
        request_id="req-1",
        trace_id="trace-1",
        job_id=None,
        route="/v1/extract",
        model_id="m1",
        params_json={
            "schema_id": "receipt_v1",
            "cache": False,
            "repair": True,
            "requested_max_new_tokens": 256,
        },
        prompt="receipt text",
        output='{"merchant":"Cafe"}',
        status_code=200,
        cached=False,
        error_code=None,
        error_stage=None,
        latency_ms=12.5,
        prompt_tokens=10,
        completion_tokens=4,
    )

    case = build_replay_case_from_inference_log(log)
    assert case["case_id"] == "log:7"
    assert case["replay_ready"] is True
    assert case["request"]["schema_id"] == "receipt_v1"
    assert case["request"]["text"] == "receipt text"
    assert case["expectation"]["status"] == "succeeded"
    assert case["expectation"]["output"] == {"merchant": "Cafe"}


def test_build_replay_case_from_trace_without_log_marks_missing_text():
    detail = TraceDetail(
        trace_id="trace-fail-1",
        status="failed",
        root_route="/v1/extract",
        request_kind="sync_extract",
        job_id=None,
        model_id=None,
        started_at=datetime(2026, 3, 27, tzinfo=UTC),
        finished_at=datetime(2026, 3, 27, tzinfo=UTC),
        events=[
            TraceEventView(
                created_at=datetime(2026, 3, 27, tzinfo=UTC),
                event_name="extract.accepted",
                route="/v1/extract",
                stage="start",
                status="accepted",
                request_id="req-1",
                job_id=None,
                model_id=None,
                details={"schema_id": "receipt_v1", "cache": False, "repair": True},
            ),
            TraceEventView(
                created_at=datetime(2026, 3, 27, tzinfo=UTC),
                event_name="extract.failed",
                route="/v1/extract",
                stage="assessed_gate",
                status="failed",
                request_id="req-1",
                job_id=None,
                model_id=None,
                details={"error_code": "capability_not_supported", "error_stage": "assessed_gate"},
            ),
        ],
    )

    case = build_replay_case_from_trace(detail, inference_logs=[])
    assert case["case_id"] == "trace:trace-fail-1"
    assert case["replay_ready"] is False
    assert "text" in case["missing_fields"]
    assert case["expectation"]["status"] == "failed"
    assert case["expectation"]["error_code"] == "capability_not_supported"


def test_build_regression_replay_manifest():
    manifest = build_regression_replay_manifest(
        source={"kind": "trace", "trace_id": "trace-1"},
        cases=[{"case_id": "trace:trace-1", "route": "/v1/extract"}],
    )

    assert manifest["schema_version"] == REGRESSION_REPLAY_MANIFEST_VERSION
    assert manifest["source"]["trace_id"] == "trace-1"
    assert manifest["cases"][0]["case_id"] == "trace:trace-1"
