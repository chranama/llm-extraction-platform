from __future__ import annotations

from pathlib import Path

import pytest

from llm_policy.cli import main

pytestmark = pytest.mark.integration


def test_model_onboarding_evaluate_allow_returns_zero(
    tmp_path: Path, write_thresholds_root, write_summary, allow_payload
):
    thresholds_root = write_thresholds_root(tmp_path / "thresholds")
    run_dir = write_summary(
        tmp_path / "run_eval_allow",
        {**allow_payload, "run_id": "run_eval_allow", "run_dir": str(tmp_path / "run_eval_allow")},
    )

    rc = main(
        [
            "model-onboarding",
            "evaluate",
            "--eval-run-dir",
            str(run_dir),
            "--threshold-profile",
            "extract/default",
            "--thresholds-root",
            str(thresholds_root),
            "--quiet",
        ]
    )
    assert rc == 0


def test_model_onboarding_evaluate_deny_returns_two(
    tmp_path: Path, write_thresholds_root, write_summary, deny_payload
):
    thresholds_root = write_thresholds_root(tmp_path / "thresholds")
    run_dir = write_summary(
        tmp_path / "run_eval_deny",
        {**deny_payload, "run_id": "run_eval_deny", "run_dir": str(tmp_path / "run_eval_deny")},
    )

    rc = main(
        [
            "model-onboarding",
            "evaluate",
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


def test_model_onboarding_apply_modifies_models_yaml_and_is_idempotent(
    tmp_path: Path, write_thresholds_root, write_summary, write_models_yaml, allow_payload
):
    thresholds_root = write_thresholds_root(tmp_path / "thresholds")
    run_dir = write_summary(
        tmp_path / "run_apply",
        {**allow_payload, "run_id": "run_apply", "run_dir": str(tmp_path / "run_apply")},
    )
    models_yaml = write_models_yaml(tmp_path / "models.yaml")

    rc1 = main(
        [
            "model-onboarding",
            "apply",
            "--models-yaml",
            str(models_yaml),
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
    assert rc1 == 0
    first_text = models_yaml.read_text(encoding="utf-8")
    assert "extract: true" in first_text.lower()
    assert "assessment:" in first_text

    rc2 = main(
        [
            "model-onboarding",
            "apply",
            "--models-yaml",
            str(models_yaml),
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
    assert rc2 == 0
    second_text = models_yaml.read_text(encoding="utf-8")
    assert "extract: true" in second_text.lower()
