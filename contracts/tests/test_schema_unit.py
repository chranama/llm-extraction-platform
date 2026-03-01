from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_contracts.schema import (
    SchemaValidationError,
    atomic_write_json_internal,
    read_json_internal,
    read_json_internal_versioned,
    validate_internal,
)


def _valid_eval_run_pointer_payload() -> dict:
    return {
        "schema_version": "eval_run_pointer_v1",
        "generated_at": "2026-02-28T00:00:00Z",
        "task": "extraction_sroie",
        "run_id": "run_123",
        "store": "fs",
        "run_dir": "/tmp/eval/run_123",
        "summary_path": "/tmp/eval/run_123/summary.json",
    }


def test_validate_internal_accepts_valid_payload() -> None:
    validate_internal("eval_run_pointer_v1.schema.json", _valid_eval_run_pointer_payload())


def test_validate_internal_raises_structured_error_for_invalid_payload() -> None:
    bad = _valid_eval_run_pointer_payload()
    bad["store"] = "invalid"

    with pytest.raises(SchemaValidationError) as exc:
        validate_internal("eval_run_pointer_v1.schema.json", bad)

    msg = str(exc.value)
    assert "eval_run_pointer_v1.schema.json" in msg
    assert "store" in msg


def test_atomic_write_json_internal_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "latest.json"
    payload = _valid_eval_run_pointer_payload()

    written = atomic_write_json_internal("eval_run_pointer_v1.schema.json", path, payload)
    assert written == path.resolve()
    assert path.read_text(encoding="utf-8").endswith("\n")

    loaded = read_json_internal("eval_run_pointer_v1.schema.json", path)
    assert loaded == payload


def test_read_json_internal_versioned_selects_schema(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    payload = _valid_eval_run_pointer_payload()
    path.write_text(json.dumps(payload), encoding="utf-8")

    out = read_json_internal_versioned(
        {"eval_run_pointer_v1": "eval_run_pointer_v1.schema.json"},
        path,
    )
    assert out["schema_version"] == "eval_run_pointer_v1"


def test_read_json_internal_versioned_rejects_unsupported_version(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    payload = _valid_eval_run_pointer_payload()
    payload["schema_version"] = "unknown_v9"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported schema_version"):
        read_json_internal_versioned(
            {"eval_run_pointer_v1": "eval_run_pointer_v1.schema.json"},
            path,
        )
