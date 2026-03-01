from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_contracts.runtime.eval_result_row import (
    parse_eval_result_row,
    read_results_jsonl,
)
from llm_contracts.runtime.eval_run_pointer import (
    build_eval_run_pointer_payload_v1,
    read_eval_run_pointer,
    write_eval_run_pointer,
)
from llm_contracts.runtime.eval_run_summary import (
    build_eval_run_summary_payload_v1,
    parse_eval_run_summary,
    read_eval_run_summary,
    write_eval_run_summary,
)


def test_eval_run_pointer_build_write_read_roundtrip(tmp_path: Path) -> None:
    payload = build_eval_run_pointer_payload_v1(
        task="extraction_sroie",
        run_id="run_ptr",
        store="fs",
        run_dir=str(tmp_path / "eval" / "run_ptr"),
        summary_path=str(tmp_path / "eval" / "run_ptr" / "summary.json"),
    )
    out = tmp_path / "latest.json"
    write_eval_run_pointer(out, payload)

    snap = read_eval_run_pointer(out)
    assert snap.ok is True
    assert snap.schema_version == "eval_run_pointer_v1"
    assert snap.task == "extraction_sroie"
    assert snap.run_id == "run_ptr"


def test_eval_run_summary_builder_emits_v2_payload_with_deployment() -> None:
    payload = build_eval_run_summary_payload_v1(
        task="extraction_sroie",
        run_id="run_sum",
        run_dir="/tmp/eval/run_sum",
        passed=True,
        metrics={"schema_validity_rate": 99.0},
        deployment_key="dep-a",
        deployment={"mode": "full"},
    )
    assert payload["schema_version"] == "eval_run_summary_v2"
    assert payload["deployment_key"] == "dep-a"

    snap = parse_eval_run_summary(payload)
    assert snap.ok is True
    assert snap.schema_version == "eval_run_summary_v2"
    assert snap.deployment_key == "dep-a"
    assert snap.deployment == {"mode": "full"}


def test_eval_run_summary_parse_accepts_v1_payload() -> None:
    payload = {
        "schema_version": "eval_run_summary_v1",
        "generated_at": "2026-02-28T00:00:00Z",
        "task": "extraction_sroie",
        "run_id": "run_v1",
        "run_dir": "/tmp/eval/run_v1",
        "passed": False,
        "metrics": {},
    }
    snap = parse_eval_run_summary(payload)
    assert snap.schema_version == "eval_run_summary_v1"
    assert snap.deployment_key is None
    assert snap.deployment is None


def test_eval_run_summary_read_fail_closed_on_invalid_file(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    path.write_text("{}", encoding="utf-8")

    snap = read_eval_run_summary(path)
    assert snap.ok is False
    assert snap.error and "eval_run_summary_parse_error" in snap.error


def test_eval_run_summary_write_then_read(tmp_path: Path) -> None:
    payload = build_eval_run_summary_payload_v1(
        task="extraction_sroie",
        run_id="run_roundtrip",
        run_dir=str(tmp_path / "eval" / "run_roundtrip"),
        passed=True,
        metrics={"k": "v"},
    )
    path = tmp_path / "summary.json"
    write_eval_run_summary(path, payload)

    snap = read_eval_run_summary(path)
    assert snap.ok is True
    assert snap.run_id == "run_roundtrip"
    assert snap.metrics == {"k": "v"}


def test_eval_result_row_parse_accepts_v2_and_v1() -> None:
    row_v2 = {
        "schema_version": "eval_result_row_v2",
        "task": "extraction_sroie",
        "run_id": "run_rows",
        "ok": False,
        "deployment_key": "dep1",
        "deployment": {"mode": "full"},
    }
    out_v2 = parse_eval_result_row(row_v2)
    assert out_v2.ok is True
    assert out_v2.schema_version == "eval_result_row_v2"
    assert out_v2.deployment_key == "dep1"

    row_v1 = {
        "schema_version": "eval_result_row_v1",
        "task": "extraction_sroie",
        "run_id": "run_rows",
        "ok": True,
    }
    out_v1 = parse_eval_result_row(row_v1)
    assert out_v1.schema_version == "eval_result_row_v1"
    assert out_v1.deployment_key is None
    assert out_v1.deployment is None


def test_read_results_jsonl_lenient_counts_invalid_rows(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "schema_version": "eval_result_row_v2",
                        "task": "extraction_sroie",
                        "run_id": "run_jsonl",
                        "ok": True,
                    }
                ),
                json.dumps({"schema_version": "eval_result_row_v2", "task": "x", "run_id": "y"}),
                "not-json",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows, stats = read_results_jsonl(path, strict=False)
    assert len(rows) == 1
    assert stats.total_lines == 3
    assert stats.ok_rows == 1
    assert stats.skipped_invalid == 1
    assert stats.parse_errors == 1


def test_read_results_jsonl_strict_raises_on_bad_json(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    path.write_text('{"schema_version":"eval_result_row_v2","task":"t","run_id":"r","ok":true}\nnot-json\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSON on line 2"):
        read_results_jsonl(path, strict=True)
