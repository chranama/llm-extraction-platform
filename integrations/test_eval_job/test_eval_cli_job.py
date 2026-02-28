from __future__ import annotations

from pathlib import Path

import pytest

from integrations.lib.eval_job import read_json, read_jsonl, run_eval_cli

pytestmark = [pytest.mark.eval_job]


def _assert_summary_v2(summary: dict, *, task: str, run_dir: str) -> None:
    assert summary["schema_version"] == "eval_run_summary_v2"
    assert summary["task"] == task
    assert isinstance(summary.get("run_id"), str) and summary["run_id"]
    assert summary["run_dir"] == run_dir


def _assert_result_row_v2(row: dict, *, task: str, run_id: str) -> None:
    assert row["schema_version"] == "eval_result_row_v2"
    assert row["task"] == task
    assert row["run_id"] == run_id
    assert isinstance(row["ok"], bool)


def test_eval_job_unreachable_base_url_writes_error_artifacts(tmp_path: Path):
    outdir = tmp_path / "eval_results"
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

    if res.returncode != 0 and (
        "fiftyone.core.service.ServiceListenTimeout" in res.stderr
        or "Subprocess" in res.stderr and "mongod" in res.stderr
    ):
        pytest.skip("local FiftyOne/mongod is unavailable in this environment")

    assert res.returncode == 0, f"stdout:\n{res.stdout}\n\nstderr:\n{res.stderr}"

    summary = read_json(outdir / "summary.json")
    _assert_summary_v2(summary, task="extraction_sroie", run_dir=str(outdir))

    rows = read_jsonl(outdir / "results.jsonl")
    assert len(rows) == 1
    row = rows[0]
    _assert_result_row_v2(row, task=summary["task"], run_id=summary["run_id"])
    assert row["status_code"] != 200
    assert isinstance(row.get("error_code"), str) and row["error_code"]

    p = read_json(pointer)
    assert p["run_dir"] == str(outdir)


@pytest.mark.requires_api_key
@pytest.mark.server_live
def test_eval_job_live_success_artifacts(
    require_api_key: bool,
    base_url: str,
    api_key: str,
    sync_probe,
    tmp_path: Path,
):
    if int(sync_probe("/healthz")) == 0:
        pytest.skip("live server not reachable")

    outdir = tmp_path / "eval_results_live"
    pointer = tmp_path / "eval_out" / "latest.json"

    res = run_eval_cli(
        args=[
            "--task",
            "extraction_sroie",
            "--base-url",
            base_url,
            "--api-key",
            api_key,
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
    _assert_summary_v2(summary, task="extraction_sroie", run_dir=str(outdir))
    assert summary.get("metrics", {}).get("base_url") == base_url

    rows = read_jsonl(outdir / "results.jsonl")
    assert len(rows) == 1
    row = rows[0]
    _assert_result_row_v2(row, task=summary["task"], run_id=summary["run_id"])
    assert row.get("status_code") == 200

    p = read_json(pointer)
    assert p["task"] == "extraction_sroie"
    assert p["run_dir"] == str(outdir)
    assert p["summary_path"] == str(outdir / "summary.json")
