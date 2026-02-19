# eval/src/llm_eval/io/run_artifacts.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union, cast

from llm_contracts.runtime.eval_result_row import EVAL_RESULT_ROW_SCHEMA, parse_eval_result_row
from llm_contracts.runtime.eval_run_summary import EVAL_RUN_SUMMARY_SCHEMA, parse_eval_run_summary
from llm_contracts.schema import validate_internal

Pathish = Union[str, Path]


def _utc_now_iso() -> str:
    # RFC3339-ish, seconds precision, Z suffix
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _atomic_write_text(path: Path, text: str) -> None:
    _ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _atomic_write_json(path: Path, obj: Any) -> None:
    # Intentionally do not use default=str; fail loudly if not JSON-safe.
    text = json.dumps(obj, ensure_ascii=False, indent=2) + "\n"
    _atomic_write_text(path, text)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    _ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            if isinstance(r, dict):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def default_outdir(root: str, task: str, run_id: str) -> str:
    return str(Path(root) / task / run_id)


@dataclass(frozen=True)
class EvalRunPaths:
    outdir: Path
    summary_json: Path
    results_jsonl: Path
    report_txt: Path
    report_md: Path
    config_json: Path


def make_run_paths(outdir: Pathish) -> EvalRunPaths:
    d = Path(outdir)
    return EvalRunPaths(
        outdir=d,
        summary_json=d / "summary.json",
        results_jsonl=d / "results.jsonl",
        report_txt=d / "report.txt",
        report_md=d / "report.md",
        config_json=d / "config.json",
    )


def _validate_eval_summary_payload(payload: dict[str, Any]) -> None:
    """
    Validate the eval run summary against the canonical internal schema.

    We validate twice:
      1) json-schema validation (fast, authoritative)
      2) parse into the stable dataclass (catches version mismatches, etc.)
    """
    validate_internal(EVAL_RUN_SUMMARY_SCHEMA, payload)
    # Also ensures schema_version matches expected value in the parser
    parse_eval_run_summary(payload)


def _validate_eval_result_row_payload(payload: dict[str, Any]) -> None:
    """
    Validate a single results.jsonl row against the canonical schema + parser.
    """
    validate_internal(EVAL_RESULT_ROW_SCHEMA, payload)
    parse_eval_result_row(payload)


def write_eval_run_artifacts(
    *,
    outdir: Pathish,
    summary: dict[str, Any],
    results: list[dict[str, Any]],
    report_txt: str,
    report_md: str,
    returned_config: Optional[dict[str, Any]] = None,
) -> EvalRunPaths:
    """
    Canonical persistence for an eval run directory.

    Contract:
      - summary MUST include 'task' and 'run_id' before writing.
      - We inject 'run_dir' (absolute-ish string path) before validation/write.
      - summary.json is validated against schemas/internal/eval_run_summary_v1.schema.json
        via llm_contracts.runtime.eval_run_summary.
      - each row in results.jsonl is validated against schemas/internal/eval_result_row_v1.schema.json
        via llm_contracts.runtime.eval_result_row.
      - Atomic writes for summary/report/config so downstream consumers never see partial JSON.
      - results.jsonl is written to tmp then replaced (atomic at file level).
    """
    paths = make_run_paths(outdir)

    # Ensure run_dir present before writing summary.json (tests + downstream consumers depend on this)
    summary_payload = dict(summary)
    summary_payload["run_dir"] = str(paths.outdir)

    # Validate summary contract (fail loudly if not schema-safe)
    _validate_eval_summary_payload(cast(dict[str, Any], summary_payload))

    # Validate result rows contract (fail loudly if any row is invalid)
    if results:
        for i, row in enumerate(results, start=1):
            if not isinstance(row, dict):
                raise TypeError(f"results[{i}] must be a dict, got {type(row).__name__}")
            _validate_eval_result_row_payload(cast(dict[str, Any], row))

    # Persist after validation passes
    _atomic_write_json(paths.summary_json, summary_payload)

    if results:
        _write_jsonl(paths.results_jsonl, results)

    _atomic_write_text(paths.report_txt, report_txt)
    _atomic_write_text(paths.report_md, report_md)

    if isinstance(returned_config, dict):
        _atomic_write_json(paths.config_json, returned_config)

    return paths


def default_eval_out_pointer_path() -> Path:
    """
    Conventional host path for a pointer artifact that indicates the latest run.
    Mirrors policy_out/latest.json style, but for eval.

    NOTE: You now have a *separate* internal contract for eval pointers:
      - contracts/src/llm_contracts/runtime/eval_run_pointer.py
      - schema: eval_run_pointer_v1.schema.json

    This function remains for backward compatibility with existing eval code.
    """
    return Path("eval_out") / "latest.json"


def write_eval_latest_pointer(
    *,
    pointer_path: Pathish,
    task: str,
    run_id: str,
    run_dir: Pathish,
    summary_path: Optional[Pathish] = None,
    extra: Optional[dict[str, Any]] = None,
) -> Path:
    """
    Back-compat pointer writer used by eval/.

    If you want to fully align to contracts/, switch callers to
    llm_contracts.runtime.eval_run_pointer.build_eval_run_pointer_payload_v1
    + write_eval_run_pointer. For now we keep this minimal-diff.

    Writes a tiny pointer artifact for "latest eval run".
    """
    p = Path(pointer_path)
    payload: dict[str, Any] = {
        "schema_version": "eval_pointer_v1",  # legacy (NOT eval_run_pointer_v1)
        "generated_at": _utc_now_iso(),
        "task": task,
        "run_id": run_id,
        "run_dir": str(Path(run_dir)),
    }
    if summary_path is not None:
        payload["summary_path"] = str(Path(summary_path))
    if extra:
        payload["extra"] = dict(extra)

    _atomic_write_json(p, payload)
    return p