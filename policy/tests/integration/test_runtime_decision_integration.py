from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_policy.cli import main

pytestmark = pytest.mark.integration


def test_runtime_decision_extract_only_allow(
    tmp_path: Path, write_thresholds_root, write_summary, allow_payload
):
    thresholds_root = write_thresholds_root(tmp_path / "thresholds")
    run_dir = write_summary(
        tmp_path / "run_allow", {**allow_payload, "run_dir": str(tmp_path / "run_allow")}
    )

    rc = main(
        [
            "runtime-decision",
            "--pipeline",
            "extract_only",
            "--run-dir",
            str(run_dir),
            "--threshold-profile",
            "extract/default",
            "--thresholds-root",
            str(thresholds_root),
            "--report",
            "text",
            "--no-write-artifact",
        ]
    )
    assert rc == 0


def test_runtime_decision_extract_only_deny(
    tmp_path: Path, write_thresholds_root, write_summary, deny_payload
):
    thresholds_root = write_thresholds_root(tmp_path / "thresholds")
    run_dir = write_summary(
        tmp_path / "run_deny", {**deny_payload, "run_dir": str(tmp_path / "run_deny")}
    )

    rc = main(
        [
            "runtime-decision",
            "--pipeline",
            "extract_only",
            "--run-dir",
            str(run_dir),
            "--threshold-profile",
            "extract/default",
            "--thresholds-root",
            str(thresholds_root),
            "--report",
            "text",
            "--no-write-artifact",
        ]
    )
    assert rc == 2


def test_runtime_decision_generate_clamp_only_returns_zero_without_eval(
    tmp_path: Path, write_thresholds_root
):
    thresholds_root = write_thresholds_root(tmp_path / "thresholds")

    rc = main(
        [
            "runtime-decision",
            "--pipeline",
            "generate_clamp_only",
            "--thresholds-root",
            str(thresholds_root),
            "--report",
            "md",
            "--no-write-artifact",
            "--no-generate-clamp",
        ]
    )
    assert rc == 0


def test_runtime_decision_extract_plus_generate_clamp_runs(
    tmp_path: Path, write_thresholds_root, write_summary, allow_payload
):
    thresholds_root = write_thresholds_root(tmp_path / "thresholds")
    run_dir = write_summary(
        tmp_path / "run_both",
        {**allow_payload, "run_id": "run_both", "run_dir": str(tmp_path / "run_both")},
    )

    rc = main(
        [
            "runtime-decision",
            "--pipeline",
            "extract_plus_generate_clamp",
            "--run-dir",
            str(run_dir),
            "--threshold-profile",
            "extract/default",
            "--thresholds-root",
            str(thresholds_root),
            "--report",
            "text",
            "--no-write-artifact",
            "--no-generate-clamp",
        ]
    )
    assert rc == 0


def test_runtime_decision_report_out_writes_file(
    tmp_path: Path, write_thresholds_root, write_summary, allow_payload
):
    thresholds_root = write_thresholds_root(tmp_path / "thresholds")
    run_dir = write_summary(
        tmp_path / "run_report",
        {**allow_payload, "run_id": "run_report", "run_dir": str(tmp_path / "run_report")},
    )
    report_out = tmp_path / "policy_report.md"

    rc = main(
        [
            "runtime-decision",
            "--pipeline",
            "extract_only",
            "--run-dir",
            str(run_dir),
            "--threshold-profile",
            "extract/default",
            "--thresholds-root",
            str(thresholds_root),
            "--report",
            "md",
            "--report-out",
            str(report_out),
            "--no-write-artifact",
        ]
    )
    assert rc == 0
    assert report_out.exists()
    assert "Provenance" in report_out.read_text(encoding="utf-8")


def test_runtime_decision_writes_artifact_extract_only(
    tmp_path: Path,
    write_thresholds_root,
    write_summary,
    allow_payload,
):
    thresholds_root = write_thresholds_root(tmp_path / "thresholds")
    run_dir = write_summary(
        tmp_path / "run_artifact",
        {**allow_payload, "run_id": "run_artifact", "run_dir": str(tmp_path / "run_artifact")},
    )
    artifact_out = tmp_path / "policy_out" / "latest.json"

    rc = main(
        [
            "runtime-decision",
            "--pipeline",
            "extract_only",
            "--run-dir",
            str(run_dir),
            "--threshold-profile",
            "extract/default",
            "--thresholds-root",
            str(thresholds_root),
            "--artifact-out",
            str(artifact_out),
            "--report",
            "text",
        ]
    )
    assert rc == 0
    payload = json.loads(artifact_out.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "policy_decision_v2"
    assert payload["pipeline"] == "extract_only"
    assert payload["enable_extract"] is True


def test_runtime_decision_writes_artifact_generate_clamp_only(
    tmp_path: Path, write_thresholds_root
):
    thresholds_root = write_thresholds_root(tmp_path / "thresholds")
    artifact_out = tmp_path / "policy_out" / "gen_latest.json"

    rc = main(
        [
            "runtime-decision",
            "--pipeline",
            "generate_clamp_only",
            "--thresholds-root",
            str(thresholds_root),
            "--artifact-out",
            str(artifact_out),
            "--report",
            "md",
            "--no-generate-clamp",
        ]
    )
    assert rc == 0
    payload = json.loads(artifact_out.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "policy_decision_v2"
    assert payload["pipeline"] == "generate_clamp_only"
    assert payload["enable_extract"] is None
    assert payload["ok"] is True


def test_runtime_decision_latest_pointer_invalid_fails_closed(
    tmp_path: Path, write_thresholds_root, monkeypatch: pytest.MonkeyPatch
):
    thresholds_root = write_thresholds_root(tmp_path / "thresholds")
    bad_latest = tmp_path / "latest.json"
    bad_latest.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("EVAL_LATEST_PATH", str(bad_latest))

    rc = main(
        [
            "runtime-decision",
            "--pipeline",
            "extract_only",
            "--run-dir",
            "latest",
            "--threshold-profile",
            "extract/default",
            "--thresholds-root",
            str(thresholds_root),
            "--report",
            "text",
            "--no-write-artifact",
        ]
    )
    assert rc == 2
