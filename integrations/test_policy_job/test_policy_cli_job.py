from __future__ import annotations

import json
from pathlib import Path

import pytest

from integrations.lib.policy_job import read_json, run_policy_cli

pytestmark = [pytest.mark.policy_job]


def _write_thresholds_root(root: Path) -> Path:
    extract = root / "extract"
    extract.mkdir(parents=True, exist_ok=True)
    (extract / "default.yaml").write_text(
        "\n".join(
            [
                "version: v1",
                "metrics:",
                "  schema_validity_rate:",
                "    min: 95.0",
                "  required_present_rate:",
                "    min: 95.0",
                "  doc_required_exact_match_rate:",
                "    min: 80.0",
                "params:",
                "  min_n_total: 1",
                "  min_n_for_point_estimate: 1",
                "",
            ]
        ),
        encoding="utf-8",
    )

    generate = root / "generate"
    generate.mkdir(parents=True, exist_ok=True)
    (generate / "portable.yaml").write_text(
        "\n".join(
            [
                "min_requests: 10",
                "error_rate:",
                "  threshold: 0.02",
                "  cap: 128",
                "latency_p95_ms:",
                "  steps:",
                "    1000: 256",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return root


def _write_eval_run(run_dir: Path, payload: dict) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return run_dir


def _assert_policy_decision_v2(payload: dict, *, pipeline: str) -> None:
    assert payload["schema_version"] == "policy_decision_v2"
    assert isinstance(payload["policy"], str) and payload["policy"]
    assert payload["pipeline"] == pipeline
    assert payload["status"] in {"allow", "deny", "unknown"}
    assert isinstance(payload["ok"], bool)
    assert isinstance(payload["contract_errors"], int)
    assert isinstance(payload.get("warnings", []), list)
    assert isinstance(payload.get("reasons", []), list)


def test_policy_job_extract_only_allow_writes_v2_artifact(tmp_path: Path):
    thresholds_root = _write_thresholds_root(tmp_path / "thresholds")
    run_dir = _write_eval_run(
        tmp_path / "run_allow",
        {
            "schema_version": "eval_run_summary_v2",
            "task": "extraction_sroie",
            "run_id": "run_allow",
            "run_dir": str(tmp_path / "run_allow"),
            "n_total": 100,
            "n_ok": 95,
            "schema_validity_rate": 99.0,
            "required_present_rate": 99.0,
            "doc_required_exact_match_rate": 95.0,
            "field_exact_match_rate": {"total": 100.0},
            "deployment_key": "dep1",
            "deployment": {"name": "int-test"},
        },
    )
    artifact_out = tmp_path / "policy_out" / "latest.json"

    res = run_policy_cli(
        args=[
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
    assert res.returncode == 0, f"stdout:\n{res.stdout}\n\nstderr:\n{res.stderr}"

    payload = read_json(artifact_out)
    _assert_policy_decision_v2(payload, pipeline="extract_only")
    assert payload["enable_extract"] is True
    assert payload["eval_run_dir"] == str(run_dir)
    assert payload["eval_task"] == "extraction_sroie"
    assert payload["eval_run_id"] == "run_allow"


def test_policy_job_extract_only_deny_writes_v2_artifact(tmp_path: Path):
    thresholds_root = _write_thresholds_root(tmp_path / "thresholds")
    run_dir = _write_eval_run(
        tmp_path / "run_deny",
        {
            "schema_version": "eval_run_summary_v2",
            "task": "extraction_sroie",
            "run_id": "run_deny",
            "run_dir": str(tmp_path / "run_deny"),
            "n_total": 100,
            "n_ok": 5,
            "schema_validity_rate": 10.0,
            "required_present_rate": 10.0,
            "doc_required_exact_match_rate": 10.0,
            "deployment_key": "dep1",
            "deployment": {"name": "int-test"},
        },
    )
    artifact_out = tmp_path / "policy_out" / "fail_closed.json"

    res = run_policy_cli(
        args=[
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
        ],
    )
    assert res.returncode == 2, f"stdout:\n{res.stdout}\n\nstderr:\n{res.stderr}"

    payload = read_json(artifact_out)
    _assert_policy_decision_v2(payload, pipeline="extract_only")
    assert payload["ok"] is False
    assert payload["status"] == "deny"
    assert payload["enable_extract"] is False
    assert payload["contract_errors"] == 0
    assert payload["eval_run_dir"] == str(run_dir)


def test_policy_job_generate_clamp_only_writes_v2_artifact(tmp_path: Path):
    thresholds_root = _write_thresholds_root(tmp_path / "thresholds")
    artifact_out = tmp_path / "policy_out" / "generate_only.json"

    res = run_policy_cli(
        args=[
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
    assert res.returncode == 0, f"stdout:\n{res.stdout}\n\nstderr:\n{res.stderr}"

    payload = read_json(artifact_out)
    _assert_policy_decision_v2(payload, pipeline="generate_clamp_only")
    assert payload["enable_extract"] is None
    assert payload["ok"] is True
