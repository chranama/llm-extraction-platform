from __future__ import annotations

import argparse
from pathlib import Path

import pytest

import simulations.artifacts.main as artifacts_main


def _rt(tmp_path: Path, *, dry_run: bool = False):
    return argparse.Namespace(
        repo_root=tmp_path,
        dry_run=dry_run,
        policy_out_path=tmp_path / "policy_out" / "latest.json",
        slo_out_path=tmp_path / "slo_out" / "generate" / "latest.json",
        eval_out_path=tmp_path / "eval_out" / "extract" / "latest.json",
    )


def test_build_extract_only_policy_payload_success() -> None:
    payload = artifacts_main._build_extract_only_policy_payload(
        enable_extract=True,
        thresholds_profile="default",
        eval_run_dir="/tmp/run",
    )
    assert payload["schema_version"] == "policy_decision_v2"
    assert payload["pipeline"] == "extract_only"
    assert payload["enable_extract"] is True


def test_build_extract_only_policy_payload_rejects_bad_status() -> None:
    with pytest.raises(artifacts_main.SimError, match="status must be one of"):
        artifacts_main._build_extract_only_policy_payload(
            enable_extract=True,
            thresholds_profile="default",
            eval_run_dir="/tmp/run",
            status="bad",
        )


def test_artifacts_write_policy_allow_clamp_requires_positive_cap(tmp_path: Path) -> None:
    rt = _rt(tmp_path, dry_run=False)
    args = argparse.Namespace(
        fixture="allow_clamp",
        cap=None,
        generate_thresholds_profile="default",
        model_id=None,
        out=None,
    )
    with pytest.raises(artifacts_main.SimError, match="--cap must be a positive integer"):
        artifacts_main._artifacts_write_policy(rt, args)


def test_artifacts_demo_eval_requires_run_id(tmp_path: Path) -> None:
    rt = _rt(tmp_path, dry_run=False)
    args = argparse.Namespace(
        fixture="pass",
        run_id="",
        model_id=None,
        schema_id="ticket_v1",
        thresholds_profile="default",
    )
    with pytest.raises(artifacts_main.SimError, match="--run-id is required"):
        artifacts_main._artifacts_demo_eval(rt, args)


def test_artifacts_demo_eval_dry_run_does_not_write(tmp_path: Path) -> None:
    rt = _rt(tmp_path, dry_run=True)
    args = argparse.Namespace(
        fixture="pass",
        run_id="demo_run",
        model_id=None,
        schema_id="ticket_v1",
        thresholds_profile="default",
    )
    rc = artifacts_main._artifacts_demo_eval(rt, args)
    assert rc == 0
    assert not (tmp_path / "results").exists()


def test_artifacts_write_slo_dry_run(tmp_path: Path) -> None:
    rt = _rt(tmp_path, dry_run=True)
    args = argparse.Namespace(
        fixture="good",
        model_id="m1",
        window_seconds=300,
        out=None,
    )
    rc = artifacts_main._artifacts_write_slo(rt, args)
    assert rc == 0
    assert not rt.slo_out_path.exists()
