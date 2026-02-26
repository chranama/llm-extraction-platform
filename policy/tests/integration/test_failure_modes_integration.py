from __future__ import annotations

from pathlib import Path

import pytest

from llm_policy.cli import main

pytestmark = pytest.mark.integration


def test_runtime_decision_missing_summary_fails_closed(tmp_path: Path, write_thresholds_root):
    thresholds_root = write_thresholds_root(tmp_path / "thresholds")
    missing_run = tmp_path / "missing_run"
    missing_run.mkdir(parents=True, exist_ok=True)

    rc = main(
        [
            "runtime-decision",
            "--pipeline",
            "extract_only",
            "--run-dir",
            str(missing_run),
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


def test_runtime_decision_missing_deployment_provenance_denies(
    tmp_path: Path, write_thresholds_root, write_summary, allow_payload
):
    thresholds_root = write_thresholds_root(tmp_path / "thresholds")
    run_dir = write_summary(
        tmp_path / "run_missing_dep",
        {
            **allow_payload,
            "run_id": "run_missing_dep",
            "run_dir": str(tmp_path / "run_missing_dep"),
            "deployment_key": "",
            "deployment": {},
        },
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


def test_model_onboarding_apply_missing_models_yaml_returns_two(
    tmp_path: Path, write_thresholds_root, write_summary, allow_payload
):
    thresholds_root = write_thresholds_root(tmp_path / "thresholds")
    run_dir = write_summary(
        tmp_path / "run_missing_models",
        {
            **allow_payload,
            "run_id": "run_missing_models",
            "run_dir": str(tmp_path / "run_missing_models"),
        },
    )
    missing_models = tmp_path / "missing_models.yaml"

    rc = main(
        [
            "model-onboarding",
            "apply",
            "--models-yaml",
            str(missing_models),
            "--model-id",
            "m1",
            "--eval-run-dir",
            str(run_dir),
            "--threshold-profile",
            "extract/default",
            "--thresholds-root",
            str(thresholds_root),
            "--quiet",
        ]
    )
    assert rc == 2
