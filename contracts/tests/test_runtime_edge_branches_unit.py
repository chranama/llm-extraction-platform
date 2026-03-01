from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_contracts.runtime.eval_result_row import read_eval_result_row_json, read_results_jsonl
from llm_contracts.runtime.eval_run_pointer import default_eval_out_path, read_eval_run_pointer
from llm_contracts.schema import (
    load_internal_schema,
    validator_for_internal,
)


def test_load_internal_schema_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_internal_schema("definitely_missing.schema.json")


def test_load_internal_schema_rejects_non_object(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    schemas_root = tmp_path / "schemas"
    internal = schemas_root / "internal"
    internal.mkdir(parents=True, exist_ok=True)
    (internal / "bad.schema.json").write_text('["not-an-object"]', encoding="utf-8")
    monkeypatch.setenv("SCHEMAS_ROOT", str(schemas_root))

    with pytest.raises(TypeError, match="schema must be a JSON object"):
        load_internal_schema("bad.schema.json")


def test_validator_for_internal_returns_cached_instance() -> None:
    v1 = validator_for_internal("eval_run_pointer_v1.schema.json")
    v2 = validator_for_internal("eval_run_pointer_v1.schema.json")
    assert v1 is v2


def test_read_eval_result_row_json_success_v2(tmp_path: Path) -> None:
    path = tmp_path / "row.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "eval_result_row_v2",
                "task": "extraction_sroie",
                "run_id": "run_json",
                "ok": True,
            }
        ),
        encoding="utf-8",
    )

    row = read_eval_result_row_json(path)
    assert row.ok is True
    assert row.schema_version == "eval_result_row_v2"
    assert row.task == "extraction_sroie"


def test_read_eval_result_row_json_fail_closed_on_invalid(tmp_path: Path) -> None:
    path = tmp_path / "row_bad.json"
    path.write_text('{"schema_version":"eval_result_row_v2"}', encoding="utf-8")

    row = read_eval_result_row_json(path)
    assert row.ok is False
    assert row.error and "eval_result_row_parse_error" in row.error


def test_read_results_jsonl_respects_max_rows(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    lines = [
        {"schema_version": "eval_result_row_v2", "task": "t", "run_id": "r", "ok": True},
        {"schema_version": "eval_result_row_v2", "task": "t", "run_id": "r", "ok": False},
        {"schema_version": "eval_result_row_v2", "task": "t", "run_id": "r", "ok": True},
    ]
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")

    rows, stats = read_results_jsonl(path, strict=False, max_rows=2)
    assert len(rows) == 2
    assert stats.ok_rows == 2
    assert stats.total_lines == 3


def test_read_results_jsonl_strict_rejects_non_object_row(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    path.write_text('{"schema_version":"eval_result_row_v2","task":"t","run_id":"r","ok":true}\n[]\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Non-object JSON on line 2"):
        read_results_jsonl(path, strict=True)


def test_read_results_jsonl_strict_missing_file_raises(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.jsonl"
    with pytest.raises(FileNotFoundError):
        read_results_jsonl(missing, strict=True)


def test_default_eval_out_path_uses_task_subdir() -> None:
    p = default_eval_out_path("extraction_sroie")
    assert str(p).endswith("eval_out/extraction_sroie/latest.json")


def test_read_eval_run_pointer_fail_closed_on_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "latest.json"
    path.write_text("{not-json", encoding="utf-8")

    snap = read_eval_run_pointer(path)
    assert snap.ok is False
    assert snap.error and "eval_run_pointer_parse_error" in snap.error
