from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from llm_policy.io.eval_runs import (
    default_eval_latest_pointer_path,
    iter_results_jsonl,
    load_eval_run_dir,
    load_summary_file,
    resolve_eval_run_dir,
)


def _write_summary(run_dir: Path, payload: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_results(run_dir: Path, rows: list[dict]) -> None:
    p = run_dir / "results.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_resolve_eval_run_dir_latest_uses_pointer(monkeypatch, tmp_path: Path) -> None:
    target = tmp_path / "eval" / "r1"
    monkeypatch.setattr(
        "llm_policy.io.eval_runs.read_eval_run_pointer",
        lambda p: SimpleNamespace(ok=True, run_dir=str(target)),
    )

    assert resolve_eval_run_dir("latest") == target


def test_resolve_eval_run_dir_latest_missing_pointer(monkeypatch) -> None:
    monkeypatch.setattr(
        "llm_policy.io.eval_runs.read_eval_run_pointer",
        lambda p: SimpleNamespace(ok=False, run_dir=None),
    )

    assert resolve_eval_run_dir("latest") == Path("__missing_eval_run_dir__")


def test_default_eval_latest_pointer_path_env(monkeypatch, tmp_path: Path) -> None:
    p = tmp_path / "ptr.json"
    monkeypatch.setenv("EVAL_LATEST_PATH", str(p))
    assert default_eval_latest_pointer_path() == p.resolve()


def test_iter_results_jsonl_reports_parse_and_validation_issues(tmp_path: Path) -> None:
    p = tmp_path / "results.jsonl"
    p.write_text(
        "\n".join(
            [
                json.dumps(
                    {"doc_id": "1", "ok": True, "deployment_key": "dep1", "deployment": {"a": 1}}
                ),
                "not-json",
                json.dumps({"ok": True}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    issues = []
    rows = list(iter_results_jsonl(p, issues=issues))

    assert len(rows) == 1
    codes = {i.code for i in issues}
    assert "results_json_parse_error" in codes
    assert "results_row_validation_error" in codes


def test_iter_results_jsonl_missing_file_emits_issue(tmp_path: Path) -> None:
    issues = []
    rows = list(iter_results_jsonl(tmp_path / "missing.jsonl", issues=issues))
    assert rows == []
    assert any(i.code == "missing_results" for i in issues)


def test_load_eval_run_dir_row_deployment_mismatch_emits_issue(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_summary(
        run_dir,
        {
            "task": "extraction_sroie",
            "run_id": "r1",
            "n_total": 1,
            "n_ok": 1,
            "deployment_key": "dep1",
            "deployment": {"provider": "openai"},
        },
    )
    _write_results(
        run_dir,
        [
            {
                "doc_id": "d1",
                "ok": True,
                "deployment_key": "dep2",
                "deployment": {"provider": "openai"},
            }
        ],
    )

    res = load_eval_run_dir(run_dir)

    assert any(i.code == "deployment_key_mismatch" for i in res.issues)


def test_load_eval_run_dir_missing_results_emits_error_when_required(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_summary(
        run_dir,
        {
            "task": "extraction_sroie",
            "run_id": "r1",
            "n_total": 0,
            "n_ok": 0,
            "deployment_key": "dep1",
            "deployment": {"provider": "openai"},
        },
    )

    res = load_eval_run_dir(run_dir, allow_missing_results=False)
    assert any(i.code == "missing_results" for i in res.issues)


def test_load_eval_run_dir_row_missing_deployment_fields_emit_issues(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_summary(
        run_dir,
        {
            "task": "extraction_sroie",
            "run_id": "r1",
            "n_total": 1,
            "n_ok": 1,
            "deployment_key": "dep1",
            "deployment": {"provider": "openai"},
        },
    )
    _write_results(run_dir, [{"doc_id": "d1", "ok": True}])

    res = load_eval_run_dir(run_dir)
    codes = {i.code for i in res.issues}
    assert "missing_row_deployment_key" in codes
    assert "missing_row_deployment" in codes


def test_load_eval_run_dir_keeps_distinct_missing_deployment_issues(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_summary(
        run_dir,
        {
            "task": "extraction_sroie",
            "run_id": "r1",
            "n_total": 0,
            "n_ok": 0,
            "deployment_key": "",
            "deployment": {},
        },
    )

    res = load_eval_run_dir(run_dir, load_results=False)

    missing_dep = [i for i in res.issues if i.code == "missing_deployment"]
    assert len(missing_dep) == 2
    assert any(i.context for i in missing_dep)
    assert any(i.context is None for i in missing_dep)


def test_load_eval_run_dir_missing_summary_returns_fail_closed_minimal_summary(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "missing"
    run_dir.mkdir(parents=True)

    res = load_eval_run_dir(run_dir, load_results=False)

    assert any(i.code == "missing_summary" for i in res.issues)
    assert res.artifact.summary.task == "unknown"
    assert res.artifact.summary.run_id == "unknown"


def test_load_summary_file_parse_and_validation_errors(tmp_path: Path) -> None:
    bad_json = tmp_path / "bad_summary.json"
    bad_json.write_text("{not-json", encoding="utf-8")
    r1 = load_summary_file(bad_json)
    assert any(i.code == "summary_json_parse_error" for i in r1.issues)

    non_obj = tmp_path / "non_obj_summary.json"
    non_obj.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    r2 = load_summary_file(non_obj)
    assert any(i.code == "summary_not_object" for i in r2.issues)

    invalid = tmp_path / "invalid_summary.json"
    invalid.write_text(
        json.dumps({"task": "x", "run_id": "r1", "n_total": -1, "deployment_key": "dep1"}),
        encoding="utf-8",
    )
    r3 = load_summary_file(invalid)
    assert any(i.code == "summary_validation_error" for i in r3.issues)
