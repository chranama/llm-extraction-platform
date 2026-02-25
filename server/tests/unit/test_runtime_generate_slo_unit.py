from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_server.io import runtime_generate_slo as slo


def test_default_generate_slo_paths(monkeypatch):
    monkeypatch.delenv("SLO_OUT_DIR", raising=False)
    assert slo.default_generate_slo_dir() == Path("slo_out") / "generate"
    assert slo.default_generate_slo_path() == (Path("slo_out") / "generate" / "latest.json")

    monkeypatch.setenv("SLO_OUT_DIR", "/tmp/slo-dir")
    assert slo.default_generate_slo_dir() == Path("/tmp/slo-dir")


def test_atomic_write_json(tmp_path: Path):
    p = tmp_path / "nested" / "out.json"
    payload = {"a": 1, "b": {"c": 2}}
    slo._atomic_write_json(p, payload)
    assert p.exists()
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data == payload


@pytest.mark.anyio
async def test_write_generate_slo_artifact(monkeypatch, tmp_path: Path):
    payload = {
        "schema_version": "runtime_generate_slo_v1",
        "generated_at": "2026-01-01T00:00:00Z",
        "window_seconds": 300,
        "window_end": "2026-01-01T00:00:00Z",
        "routes": ["/v1/generate"],
        "models": [],
        "totals": {
            "requests": {"total": 0},
            "errors": {"total": 0, "rate": 0.0, "by_status": {}, "by_code": {}},
            "latency_ms": {"p50": 0.0, "p95": 0.0, "p99": 0.0, "avg": 0.0, "max": 0.0},
            "tokens": {
                "prompt": {"avg": 0.0, "max": 0},
                "completion": {"avg": 0.0, "p95": 0.0, "max": 0},
            },
        },
    }

    async def _compute(session, *, window_seconds, routes=None, model_id=None):
        return dict(payload)

    monkeypatch.setattr(slo, "compute_window_generate_slo_contracts_payload", _compute, raising=True)
    monkeypatch.setattr(slo, "parse_generate_slo", lambda x: x, raising=True)

    out_path = tmp_path / "artifact.json"
    res = await slo.write_generate_slo_artifact(
        session=object(),
        window_seconds=300,
        routes=["/v1/generate"],
        model_id="m1",
        out_path=out_path,
    )
    assert res.ok is True
    assert Path(res.out_path) == out_path
    on_disk = json.loads(out_path.read_text(encoding="utf-8"))
    assert on_disk["schema_version"] == "runtime_generate_slo_v1"
