from __future__ import annotations

from pathlib import Path

import pytest

from tests.integration.helpers import read_json, read_jsonl, run_eval_cli

pytestmark = [pytest.mark.integration, pytest.mark.integration_contract]


def test_cli_job_unreachable_base_url_writes_typed_error_artifacts(tmp_path: Path):
    outdir = tmp_path / "results"
    pointer = tmp_path / "eval_out" / "latest.json"

    res = run_eval_cli(
        args=[
            "--task",
            "extraction_sroie",
            "--base-url",
            "http://127.0.0.1:9",
            "--api-key",
            "k",
            "--max-examples",
            "1",
            "--no-print-summary",
            "--save",
            "--outdir",
            str(outdir),
        ],
        env={"EVAL_LATEST_PATH": str(pointer)},
    )

    assert res.returncode == 0, f"stdout:\n{res.stdout}\n\nstderr:\n{res.stderr}"

    summary = read_json(outdir / "summary.json")
    assert summary["task"] == "extraction_sroie"

    rows = read_jsonl(outdir / "results.jsonl")
    assert len(rows) == 1
    assert rows[0]["status_code"] != 200
    assert isinstance(rows[0].get("error_code"), str) and rows[0]["error_code"]

    p = read_json(pointer)
    assert p["run_dir"] == str(outdir)


def test_cli_pointer_disabled_by_env(tmp_path: Path):
    outdir = tmp_path / "results"
    pointer = tmp_path / "eval_out" / "latest.json"

    res = run_eval_cli(
        args=[
            "--task",
            "extraction_sroie",
            "--base-url",
            "http://127.0.0.1:9",
            "--api-key",
            "k",
            "--max-examples",
            "1",
            "--no-print-summary",
            "--save",
            "--outdir",
            str(outdir),
        ],
        env={"EVAL_LATEST_PATH": str(pointer), "EVAL_WRITE_LATEST": "0"},
    )

    assert res.returncode == 0
    assert (outdir / "summary.json").exists()
    assert not pointer.exists()


def test_cli_missing_task_errors_nonzero():
    res = run_eval_cli(args=["--list-tasks"])
    # list-tasks should work and exit 0
    assert res.returncode == 0

    # missing --task and no --list-tasks should exit non-zero
    res2 = run_eval_cli(args=[])
    assert res2.returncode != 0
