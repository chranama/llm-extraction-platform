from __future__ import annotations

from pathlib import Path

import pytest

from llm_policy.cli import main


@pytest.fixture(autouse=True)
def _patch_cli_dependencies(monkeypatch: pytest.MonkeyPatch):
    class FakeDecision:
        def __init__(self, ok: bool = True, pipeline: str = "extract_only"):
            self.pipeline = pipeline
            self.policy = "llm_policy"
            self.enable_extract = ok
            self.status = "allow" if ok else "deny"
            self.reasons = []
            self.warnings = []
            self.metrics = {}

        def ok(self) -> bool:
            return self.enable_extract

        def model_copy(self, update=None):
            d = FakeDecision(self.enable_extract, self.pipeline)
            if update:
                for k, v in update.items():
                    setattr(d, k, v)
            return d

    monkeypatch.setattr(
        "llm_policy.cli._build_runtime_decision", lambda args: FakeDecision(True), raising=True
    )
    monkeypatch.setattr("llm_policy.cli.render_decision_text", lambda d: "TEXT\n", raising=True)
    monkeypatch.setattr("llm_policy.cli.render_decision_md", lambda d: "MD\n", raising=True)
    monkeypatch.setattr(
        "llm_policy.cli.write_policy_decision_artifact", lambda d, p: Path(p), raising=True
    )


def test_runtime_decision_text_to_stdout(capsys: pytest.CaptureFixture[str], tmp_path: Path):
    rc = main(
        [
            "runtime-decision",
            "--pipeline",
            "extract_only",
            "--run-dir",
            str(tmp_path),
            "--report",
            "text",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "== Provenance ==" in out
    assert "eval_run_dir:" in out
    assert "deployment_key:" in out
    assert out.endswith("TEXT\n")


def test_runtime_decision_md_to_file(tmp_path: Path):
    out_file = tmp_path / "out.md"
    rc = main(
        [
            "runtime-decision",
            "--pipeline",
            "extract_only",
            "--run-dir",
            str(tmp_path),
            "--report",
            "md",
            "--report-out",
            str(out_file),
        ]
    )
    assert rc == 0
    out = out_file.read_text(encoding="utf-8")
    assert "== Provenance ==" in out
    assert "eval_run_dir:" in out
    assert "deployment_key:" in out
    assert out.endswith("MD\n")


def test_runtime_decision_exit_code_2_when_not_ok(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    class FakeDecision:
        pipeline = "extract_only"
        policy = "llm_policy"
        enable_extract = False
        status = "deny"
        reasons = []
        warnings = []
        metrics = {}

        def ok(self) -> bool:
            return False

        def model_copy(self, update=None):
            return self

    monkeypatch.setattr(
        "llm_policy.cli._build_runtime_decision", lambda args: FakeDecision(), raising=True
    )

    rc = main(
        [
            "runtime-decision",
            "--pipeline",
            "extract_only",
            "--run-dir",
            str(tmp_path),
            "--report",
            "text",
        ]
    )
    assert rc == 2
