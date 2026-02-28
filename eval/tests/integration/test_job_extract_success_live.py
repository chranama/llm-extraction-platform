from __future__ import annotations

from pathlib import Path

import pytest

from tests.integration.helpers import read_json, read_jsonl, run_eval_cli

pytestmark = [pytest.mark.integration, pytest.mark.integration_live]


@pytest.mark.asyncio
async def test_cli_job_extract_success_live(
    require_live_server: None,
    integration_base_url: str,
    integration_api_key: str,
    tmp_path: Path,
):
    outdir = tmp_path / "results"
    pointer = tmp_path / "eval_out" / "latest.json"

    res = run_eval_cli(
        args=[
            "--task",
            "extraction_sroie",
            "--base-url",
            integration_base_url,
            "--api-key",
            integration_api_key,
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

    summary_p = outdir / "summary.json"
    report_txt_p = outdir / "report.txt"
    report_md_p = outdir / "report.md"

    assert summary_p.exists()
    assert report_txt_p.exists()
    assert report_md_p.exists()
    assert pointer.exists()

    summary = read_json(summary_p)
    assert summary["task"] == "extraction_sroie"
    assert isinstance(summary.get("run_id"), str) and summary["run_id"]
    assert summary["schema_version"] == "eval_run_summary_v2"
    assert summary.get("metrics", {}).get("base_url") == integration_base_url
    assert summary["run_dir"] == str(outdir)

    results_p = outdir / "results.jsonl"
    if results_p.exists():
        rows = read_jsonl(results_p)
        assert isinstance(rows, list)
        if rows:
            row = rows[0]
            assert row["schema_version"] == "eval_result_row_v2"
            assert row["task"] == summary["task"]
            assert row["run_id"] == summary["run_id"]
            assert isinstance(row["ok"], bool)

    p = read_json(pointer)
    assert p["task"] == "extraction_sroie"
    assert p["run_dir"] == str(outdir)
    assert p["summary_path"] == str(summary_p)
