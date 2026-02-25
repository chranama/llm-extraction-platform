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
    results_jsonl: Optional[Path]
    report_txt: Path
    report_md: Path
    config_json: Optional[Path]


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
    try:
        validate_internal(EVAL_RUN_SUMMARY_SCHEMA, payload)
        parse_eval_run_summary(payload)
    except FileNotFoundError:
        # Some local/dev test layouts intentionally omit internal schema files.
        return


def _validate_eval_result_row_payload(payload: dict[str, Any]) -> None:
    """
    Validate a single results.jsonl row against the canonical schema + parser.
    """
    try:
        validate_internal(EVAL_RESULT_ROW_SCHEMA, payload)
        parse_eval_result_row(payload)
    except FileNotFoundError:
        return


def _infer_deployment_from_summary_or_rows(
    *,
    summary_payload: dict[str, Any],
    results: list[dict[str, Any]],
) -> tuple[Optional[str], Optional[dict[str, Any]]]:
    """
    Best-effort: infer deployment_key + deployment snapshot.

    Precedence:
      1) summary.{deployment_key,deployment}
      2) first row.{deployment_key,deployment}
      3) None
    """
    dk = summary_payload.get("deployment_key")
    dep = summary_payload.get("deployment")

    if isinstance(dk, str) and dk.strip() and isinstance(dep, dict):
        return dk.strip(), cast(dict[str, Any], dep)

    if results:
        r0 = results[0] if isinstance(results[0], dict) else {}
        dk0 = r0.get("deployment_key")
        dep0 = r0.get("deployment")
        if isinstance(dk0, str) and dk0.strip() and isinstance(dep0, dict):
            return dk0.strip(), cast(dict[str, Any], dep0)

    # allow partials (key only / deployment only)
    dk_out: Optional[str] = dk.strip() if isinstance(dk, str) and dk.strip() else None
    dep_out: Optional[dict[str, Any]] = dict(dep) if isinstance(dep, dict) else None
    return dk_out, dep_out


def _inject_deployment_into_rows(
    *,
    results: list[dict[str, Any]],
    deployment_key: Optional[str],
    deployment: Optional[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Ensure each result row carries deployment fields (v2 contract).

    Rules:
      - Never overwrite explicit row values.
      - If summary had values and row is missing, inject them.
    """
    if not results:
        return results

    dk = (
        deployment_key.strip()
        if isinstance(deployment_key, str) and deployment_key.strip()
        else None
    )
    dep = dict(deployment) if isinstance(deployment, dict) else None

    if dk is None and dep is None:
        return results

    out: list[dict[str, Any]] = []
    for row in results:
        if not isinstance(row, dict):
            continue
        r = dict(row)
        if dk is not None and not (
            isinstance(r.get("deployment_key"), str) and str(r.get("deployment_key")).strip()
        ):
            r["deployment_key"] = dk
        if dep is not None and not isinstance(r.get("deployment"), dict):
            r["deployment"] = dep
        out.append(r)
    return out


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
      - summary.json is validated against the current eval run summary schema (v2 after upgrade).
      - each row in results.jsonl is validated against the current eval result row schema (v2 after upgrade).
      - Deployment info is expected in BOTH summary + rows (not pointer):
          - deployment_key
          - deployment (object)
        If missing from rows, we inject from the summary (best-effort).
      - Atomic writes for summary/report/config so downstream consumers never see partial JSON.
      - results.jsonl is written to tmp then replaced (atomic at file level).
    """
    paths = make_run_paths(outdir)

    # Ensure run_dir present before writing summary.json (tests + downstream consumers depend on this)
    summary_payload = dict(summary)
    summary_payload["run_dir"] = str(paths.outdir)

    # Best-effort deployment propagation:
    # - infer from summary or first row
    # - ensure summary includes it (if inferred)
    # - ensure each row includes it (if summary has it)
    inferred_dk, inferred_dep = _infer_deployment_from_summary_or_rows(
        summary_payload=summary_payload, results=results
    )

    if inferred_dk is not None and not (
        isinstance(summary_payload.get("deployment_key"), str)
        and str(summary_payload.get("deployment_key")).strip()
    ):
        summary_payload["deployment_key"] = inferred_dk

    if inferred_dep is not None and not isinstance(summary_payload.get("deployment"), dict):
        summary_payload["deployment"] = inferred_dep

    results_payload = _inject_deployment_into_rows(
        results=results,
        deployment_key=(
            summary_payload.get("deployment_key")
            if isinstance(summary_payload.get("deployment_key"), str)
            else None
        ),
        deployment=(
            summary_payload.get("deployment")
            if isinstance(summary_payload.get("deployment"), dict)
            else None
        ),
    )

    # Validate summary contract (fail loudly if not schema-safe)
    _validate_eval_summary_payload(cast(dict[str, Any], summary_payload))

    # Validate result rows contract (fail loudly if any row is invalid)
    if results_payload:
        for i, row in enumerate(results_payload, start=1):
            if not isinstance(row, dict):
                raise TypeError(f"results[{i}] must be a dict, got {type(row).__name__}")
            _validate_eval_result_row_payload(cast(dict[str, Any], row))

    # Persist after validation passes
    _atomic_write_json(paths.summary_json, summary_payload)

    wrote_results = False
    if results_payload:
        _write_jsonl(paths.results_jsonl, results_payload)
        wrote_results = True

    _atomic_write_text(paths.report_txt, report_txt)
    _atomic_write_text(paths.report_md, report_md)

    wrote_config = False
    if isinstance(returned_config, dict):
        _atomic_write_json(paths.config_json, returned_config)
        wrote_config = True

    return EvalRunPaths(
        outdir=paths.outdir,
        summary_json=paths.summary_json,
        results_jsonl=paths.results_jsonl if wrote_results else None,
        report_txt=paths.report_txt,
        report_md=paths.report_md,
        config_json=paths.config_json if wrote_config else None,
    )


def default_eval_out_pointer_path() -> Path:
    """
    Conventional host path for a pointer artifact that indicates the latest run.
    Mirrors policy_out/latest.json style, but for eval.

    NOTE:
      - The pointer is convenience-only.
      - Deployment info belongs in run summary + rows (not the pointer).
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

    Writes a tiny pointer artifact for "latest eval run".
    Deployment info is intentionally NOT stored here.
    """
    p = Path(pointer_path)
    payload: dict[str, Any] = {
        "schema_version": "eval_latest_v1",
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
