from __future__ import annotations

from pathlib import Path

import pytest

from llm_policy.cli import main


def test_model_onboarding_apply_calls_apply_model_onboarding(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    calls = {}

    class Res:
        ok = True

    def fake_apply(**kwargs):
        calls.update(kwargs)
        return Res()

    monkeypatch.setattr("llm_policy.cli.apply_model_onboarding", fake_apply, raising=True)

    rc = main(
        [
            "model-onboarding",
            "apply",
            "--models-yaml",
            str(tmp_path / "models.yaml"),
            "--model-id",
            "m1",
            "--eval-run-dir",
            str(tmp_path / "run"),
            "--threshold-profile",
            "extract/default",
            "--thresholds-root",
            str(tmp_path / "thr"),
            "--quiet",
        ]
    )
    assert rc == 0

    assert calls["model_id"] == "m1"
    assert calls["models_yaml"].endswith("models.yaml")
    assert calls["eval_run_dir"].endswith("run")
    assert calls["threshold_profile"] == "extract/default"
    assert calls["verbose"] is False


def test_model_onboarding_evaluate_exit_code_2_when_not_ok(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    class D:
        def ok(self):
            return False

    class Res:
        decision = D()

    monkeypatch.setattr(
        "llm_policy.cli.evaluate_model_onboarding", lambda **kwargs: Res(), raising=True
    )

    rc = main(
        [
            "model-onboarding",
            "evaluate",
            "--eval-run-dir",
            str(tmp_path / "run"),
            "--quiet",
        ]
    )
    assert rc == 2
