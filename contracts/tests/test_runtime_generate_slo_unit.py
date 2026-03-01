from __future__ import annotations

import json
from pathlib import Path

from llm_contracts.runtime.generate_slo import (
    parse_generate_slo,
    read_generate_slo,
    write_generate_slo,
)


def _generate_slo_payload() -> dict:
    return {
        "schema_version": "runtime_generate_slo_v1",
        "generated_at": "2026-02-28T00:00:00Z",
        "window_seconds": 300,
        "window_end": "2026-02-28T00:05:00Z",
        "routes": ["/v1/generate"],
        "models": [
            {
                "model_id": "m1",
                "requests": {"total": 10},
                "errors": {"total": 1, "rate": 0.1},
                "latency_ms": {"p50": 100, "p95": 200, "p99": 300, "avg": 120, "max": 350},
                "tokens": {
                    "prompt": {"avg": 50, "max": 120},
                    "completion": {"avg": 30, "p95": 60, "max": 90},
                },
            }
        ],
        "totals": {
            "requests": {"total": 10},
            "errors": {"total": 1, "rate": 0.1},
            "latency_ms": {"p50": 100, "p95": 200, "p99": 300, "avg": 120, "max": 350},
            "tokens": {
                "prompt": {"avg": 50, "max": 120},
                "completion": {"avg": 30, "p95": 60, "max": 90},
            },
        },
    }


def test_parse_generate_slo_extracts_totals() -> None:
    snap = parse_generate_slo(_generate_slo_payload())
    assert snap.schema_version == "runtime_generate_slo_v1"
    assert snap.total_requests == 10
    assert snap.error_rate == 0.1
    assert snap.latency_p95_ms == 200.0
    assert snap.completion_tokens_p95 == 60.0


def test_write_then_read_generate_slo_roundtrip(tmp_path: Path) -> None:
    payload = _generate_slo_payload()
    path = tmp_path / "generate_slo.json"
    write_generate_slo(path, payload)

    snap = read_generate_slo(path)
    assert snap.error is None
    assert snap.schema_version == "runtime_generate_slo_v1"
    assert snap.total_requests == 10


def test_read_generate_slo_fail_closed_on_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not-json", encoding="utf-8")

    snap = read_generate_slo(path)
    assert snap.schema_version == ""
    assert snap.error and "generate_slo_parse_error" in snap.error
    assert snap.error_rate == 1.0
