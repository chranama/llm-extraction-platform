# simulations/artifacts/eval_fixtures.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from llm_contracts.runtime.eval_run_pointer import (
    build_eval_run_pointer_payload_v1,
    default_eval_out_path,
    write_eval_run_pointer,
)
from llm_contracts.runtime.eval_run_summary import (
    EVAL_RUN_SUMMARY_SCHEMA,
    parse_eval_run_summary,
)
from llm_contracts.runtime.eval_result_row import (
    EVAL_RESULT_ROW_SCHEMA,
    parse_eval_result_row,
    read_results_jsonl,
)
from llm_contracts.schema import validate_internal

Pathish = Union[str, Path]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _atomic_write_text(path: Path, text: str) -> None:
    _ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _atomic_write_json(path: Path, obj: Any) -> None:
    text = json.dumps(obj, ensure_ascii=False, indent=2) + "\n"
    _atomic_write_text(path, text)


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    _ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


@dataclass(frozen=True)
class EvalFixturePaths:
    """
    What we write for eval fixtures:

    - run_dir: results/<task>/<run_id>/
      - summary.json (eval_run_summary_v1)
      - results.jsonl (eval_result_row_v1 per line; optional)
    - pointer_json: eval_out/<task>/latest.json (eval_run_pointer_v1)
    """
    run_dir: Path
    summary_json: Path
    results_jsonl: Path
    pointer_json: Path


def default_run_dir(repo_root: Path, *, task: str, run_id: str) -> Path:
    return (repo_root / "results" / task / run_id).resolve()


def _make_paths(*, repo_root: Path, task: str, run_id: str) -> EvalFixturePaths:
    run_dir = default_run_dir(repo_root, task=task, run_id=run_id)
    return EvalFixturePaths(
        run_dir=run_dir,
        summary_json=run_dir / "summary.json",
        results_jsonl=run_dir / "results.jsonl",
        pointer_json=default_eval_out_path(task),
    )


# -----------------------------------------------------------------------------
# Builders aligned to your schemas
# -----------------------------------------------------------------------------


def build_eval_run_summary_v1(
    *,
    task: str,
    run_id: str,
    run_dir: str,
    passed: bool,
    metrics: Dict[str, Any],
    generated_at: Optional[str] = None,
    model_id: Optional[str] = None,
    schema_id: Optional[str] = None,
    thresholds_profile: Optional[str] = None,
    thresholds_version: Optional[str] = None,
    counts: Optional[Dict[str, Any]] = None,
    warnings: Optional[List[Dict[str, Any]]] = None,
    notes: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "schema_version": "eval_run_summary_v1",
        "generated_at": generated_at or _utc_now_iso(),
        "task": str(task),
        "run_id": str(run_id),
        "run_dir": str(run_dir),
        "passed": bool(passed),
        "metrics": dict(metrics),
    }

    payload["model_id"] = model_id
    payload["schema_id"] = schema_id
    payload["thresholds_profile"] = thresholds_profile
    payload["thresholds_version"] = thresholds_version
    if isinstance(counts, dict):
        payload["counts"] = dict(counts)
    payload["warnings"] = list(warnings or [])
    payload["notes"] = dict(notes) if isinstance(notes, dict) else None

    def _drop_nullish_minlen_str(k: str) -> None:
        if payload.get(k) is None:
            payload.pop(k, None)

    for k in ("model_id", "schema_id", "thresholds_profile", "thresholds_version"):
        _drop_nullish_minlen_str(k)

    if payload.get("notes") is None:
        payload.pop("notes", None)

    validate_internal(EVAL_RUN_SUMMARY_SCHEMA, payload)
    parse_eval_run_summary(payload)
    return payload


def build_eval_result_row_v1(
    *,
    task: str,
    run_id: str,
    ok: bool,
    example_id: Optional[str] = None,
    schema_id: Optional[str] = None,
    model_id: Optional[str] = None,
    latency_ms: Optional[float] = None,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    error: Optional[Dict[str, Any]] = None,
    raw: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "schema_version": "eval_result_row_v1",
        "task": str(task),
        "run_id": str(run_id),
        "ok": bool(ok),
    }

    def _maybe_str(k: str, v: Optional[str]) -> None:
        if isinstance(v, str) and v.strip():
            payload[k] = v

    _maybe_str("example_id", example_id)
    _maybe_str("schema_id", schema_id)
    _maybe_str("model_id", model_id)

    if latency_ms is not None:
        payload["latency_ms"] = float(latency_ms)

    if prompt_tokens is not None or completion_tokens is not None:
        payload["tokens"] = {
            "prompt": int(prompt_tokens) if prompt_tokens is not None else None,
            "completion": int(completion_tokens) if completion_tokens is not None else None,
        }

    if error is not None:
        payload["error"] = dict(error)

    if raw is not None:
        payload["raw"] = dict(raw)

    if isinstance(extra, dict) and extra:
        payload.update(extra)

    validate_internal(EVAL_RESULT_ROW_SCHEMA, payload)
    parse_eval_result_row(payload)
    return payload


# -----------------------------------------------------------------------------
# Writers for run dir + pointer
# -----------------------------------------------------------------------------


def write_eval_run_dir(
    *,
    repo_root: Path,
    task: str,
    run_id: str,
    summary_payload: Dict[str, Any],
    results_rows: Optional[List[Dict[str, Any]]] = None,
) -> EvalFixturePaths:
    paths = _make_paths(repo_root=repo_root, task=task, run_id=run_id)

    summary_payload = dict(summary_payload)
    summary_payload["run_dir"] = str(paths.run_dir)

    validate_internal(EVAL_RUN_SUMMARY_SCHEMA, summary_payload)
    parse_eval_run_summary(summary_payload)

    _ensure_dir(paths.run_dir)
    _atomic_write_json(paths.summary_json, summary_payload)

    if results_rows:
        validated: list[dict[str, Any]] = []
        for row in results_rows:
            if not isinstance(row, dict):
                raise TypeError(f"results_rows items must be dicts, got {type(row).__name__}")
            validate_internal(EVAL_RESULT_ROW_SCHEMA, row)
            parse_eval_result_row(row)
            validated.append(row)

        _atomic_write_jsonl(paths.results_jsonl, validated)
        _ = read_results_jsonl(paths.results_jsonl)

    return paths


def write_eval_pointer_for_run(
    *,
    task: str,
    run_id: str,
    run_dir: Path,
    summary_path: Path,
    base_url: Optional[str] = None,
    model_override: Optional[str] = None,
    schema_id: Optional[str] = None,
    max_examples: Optional[int] = None,
    notes: Optional[Dict[str, Any]] = None,
    out_path: Optional[Pathish] = None,
) -> Path:
    pointer_path = Path(out_path).resolve() if out_path is not None else default_eval_out_path(task)
    payload = build_eval_run_pointer_payload_v1(
        task=task,
        run_id=run_id,
        store="fs",
        run_dir=str(run_dir),
        summary_path=str(summary_path),
        base_url=base_url,
        model_override=model_override,
        schema_id=schema_id,
        max_examples=max_examples,
        notes=notes,
    )
    return write_eval_run_pointer(pointer_path, payload)


# -----------------------------------------------------------------------------
# Eval convenience fixtures (formerly Demo B fixtures)
# -----------------------------------------------------------------------------


def eval_fixture_pass(
    *,
    repo_root: Path,
    task: str = "extract",
    run_id: str = "eval_pass",
    model_id: Optional[str] = None,
    schema_id: Optional[str] = "ticket_v1",
    thresholds_profile: str = "default",
) -> Tuple[EvalFixturePaths, Path]:
    """
    PASS fixture:
      - summary.passed = True
      - rows mostly ok=True
    """
    run_dir = str(default_run_dir(repo_root, task=task, run_id=run_id))

    rows = [
        build_eval_result_row_v1(
            task=task,
            run_id=run_id,
            ok=True,
            example_id="ex1",
            schema_id=schema_id,
            model_id=model_id,
            latency_ms=120.0,
            prompt_tokens=20,
            completion_tokens=15,
            raw={"fixture": "pass"},
        ),
        build_eval_result_row_v1(
            task=task,
            run_id=run_id,
            ok=True,
            example_id="ex2",
            schema_id=schema_id,
            model_id=model_id,
            latency_ms=140.0,
            prompt_tokens=22,
            completion_tokens=18,
            raw={"fixture": "pass"},
        ),
    ]

    counts = {
        "examples_total": len(rows),
        "examples_ok": sum(1 for r in rows if r.get("ok") is True),
        "examples_failed": sum(1 for r in rows if r.get("ok") is False),
    }

    summary = build_eval_run_summary_v1(
        task=task,
        run_id=run_id,
        run_dir=run_dir,
        model_id=model_id,
        schema_id=schema_id,
        passed=True,
        thresholds_profile=thresholds_profile,
        thresholds_version="fixtures",
        counts=counts,
        metrics={"extract_gate": {"passed": True}},
        warnings=[],
        notes={"fixture": "eval_pass"},
    )

    paths = write_eval_run_dir(repo_root=repo_root, task=task, run_id=run_id, summary_payload=summary, results_rows=rows)
    pointer = write_eval_pointer_for_run(
        task=task,
        run_id=run_id,
        run_dir=paths.run_dir,
        summary_path=paths.summary_json,
        notes={"fixture": "eval_pass"},
    )
    return paths, pointer


def eval_fixture_fail(
    *,
    repo_root: Path,
    task: str = "extract",
    run_id: str = "eval_fail",
    model_id: Optional[str] = None,
    schema_id: Optional[str] = "ticket_v1",
    thresholds_profile: str = "default",
) -> Tuple[EvalFixturePaths, Path]:
    """
    FAIL fixture:
      - summary.passed = False
      - rows include failures (ok=False with an error object)
    """
    run_dir = str(default_run_dir(repo_root, task=task, run_id=run_id))

    rows = [
        build_eval_result_row_v1(
            task=task,
            run_id=run_id,
            ok=False,
            example_id="ex1",
            schema_id=schema_id,
            model_id=model_id,
            latency_ms=300.0,
            prompt_tokens=25,
            completion_tokens=0,
            error={"code": "schema_validation_failed", "message": "Output did not match schema", "stage": "postprocess"},
            raw={"fixture": "fail"},
        ),
        build_eval_result_row_v1(
            task=task,
            run_id=run_id,
            ok=True,
            example_id="ex2",
            schema_id=schema_id,
            model_id=model_id,
            latency_ms=180.0,
            prompt_tokens=20,
            completion_tokens=12,
            raw={"fixture": "fail"},
        ),
    ]

    counts = {
        "examples_total": len(rows),
        "examples_ok": sum(1 for r in rows if r.get("ok") is True),
        "examples_failed": sum(1 for r in rows if r.get("ok") is False),
    }

    summary = build_eval_run_summary_v1(
        task=task,
        run_id=run_id,
        run_dir=run_dir,
        model_id=model_id,
        schema_id=schema_id,
        passed=False,
        thresholds_profile=thresholds_profile,
        thresholds_version="fixtures",
        counts=counts,
        metrics={"extract_gate": {"passed": False}},
        warnings=[],
        notes={"fixture": "eval_fail"},
    )

    paths = write_eval_run_dir(repo_root=repo_root, task=task, run_id=run_id, summary_payload=summary, results_rows=rows)
    pointer = write_eval_pointer_for_run(
        task=task,
        run_id=run_id,
        run_dir=paths.run_dir,
        summary_path=paths.summary_json,
        notes={"fixture": "eval_fail"},
    )
    return paths, pointer