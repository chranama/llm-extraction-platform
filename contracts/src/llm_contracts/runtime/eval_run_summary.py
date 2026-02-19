# contracts/src/llm_contracts/runtime/eval_run_summary.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union, cast

from llm_contracts.schema import atomic_write_json_internal, read_json_internal, validate_internal

Pathish = Union[str, Path]

EVAL_RUN_SUMMARY_SCHEMA = "eval_run_summary_v1.schema.json"
EVAL_RUN_SUMMARY_VERSION = "eval_run_summary_v1"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class EvalRunSummarySnapshot:
    """
    Stable, minimal contract for an eval run summary (summary.json).

    This is the primary cross-component artifact policy consumes for extract gating.
    Keep `raw` for forward compatibility.
    """
    ok: bool
    schema_version: str
    generated_at: str

    task: str
    run_id: str
    run_dir: str

    passed: bool
    metrics: Dict[str, Any]

    model_id: Optional[str] = None
    schema_id: Optional[str] = None
    thresholds_profile: Optional[str] = None
    thresholds_version: Optional[str] = None

    counts: Optional[Dict[str, int]] = None
    warnings: Optional[list[Dict[str, Any]]] = None
    notes: Optional[Dict[str, Any]] = None

    raw: Dict[str, Any] = None  # type: ignore[assignment]
    source_path: Optional[str] = None
    error: Optional[str] = None


def build_eval_run_summary_payload_v1(
    *,
    task: str,
    run_id: str,
    run_dir: str,
    passed: bool,
    metrics: Dict[str, Any],
    model_id: Optional[str] = None,
    schema_id: Optional[str] = None,
    thresholds_profile: Optional[str] = None,
    thresholds_version: Optional[str] = None,
    counts: Optional[Dict[str, int]] = None,
    warnings: Optional[list[Dict[str, Any]]] = None,
    notes: Optional[Dict[str, Any]] = None,
    generated_at: Optional[str] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "schema_version": EVAL_RUN_SUMMARY_VERSION,
        "generated_at": generated_at or _utc_now_iso(),
        "task": task,
        "run_id": run_id,
        "run_dir": run_dir,
        "model_id": model_id,
        "schema_id": schema_id,
        "passed": bool(passed),
        "metrics": dict(metrics or {}),
        "thresholds_profile": thresholds_profile,
        "thresholds_version": thresholds_version,
        "counts": counts,
        "warnings": warnings or [],
        "notes": notes,
    }

    # Keep compact and deterministic
    payload = {k: v for k, v in payload.items() if v is not None}

    validate_internal(EVAL_RUN_SUMMARY_SCHEMA, payload)
    return payload


def parse_eval_run_summary(payload: Dict[str, Any], *, source_path: Optional[str] = None) -> EvalRunSummarySnapshot:
    validate_internal(EVAL_RUN_SUMMARY_SCHEMA, payload)

    schema_version = str(payload["schema_version"]).strip()
    if schema_version != EVAL_RUN_SUMMARY_VERSION:
        raise ValueError(f"Unsupported eval run summary schema_version: {schema_version}")

    task = str(payload["task"]).strip()
    run_id = str(payload["run_id"]).strip()
    run_dir = str(payload["run_dir"]).strip()
    generated_at = str(payload["generated_at"]).strip()

    passed = bool(payload["passed"])
    metrics = cast(Dict[str, Any], payload.get("metrics") or {})

    counts = payload.get("counts")
    if isinstance(counts, dict):
        counts = {str(k): int(v) for k, v in counts.items() if isinstance(v, (int, float))}

    warnings = payload.get("warnings")
    if isinstance(warnings, list):
        warnings = [w for w in warnings if isinstance(w, dict)]

    notes = payload.get("notes") if isinstance(payload.get("notes"), dict) else None

    return EvalRunSummarySnapshot(
        ok=True,
        schema_version=schema_version,
        generated_at=generated_at,
        task=task,
        run_id=run_id,
        run_dir=run_dir,
        passed=passed,
        metrics=dict(metrics),
        model_id=payload.get("model_id") if isinstance(payload.get("model_id"), str) else None,
        schema_id=payload.get("schema_id") if isinstance(payload.get("schema_id"), str) else None,
        thresholds_profile=payload.get("thresholds_profile") if isinstance(payload.get("thresholds_profile"), str) else None,
        thresholds_version=payload.get("thresholds_version") if isinstance(payload.get("thresholds_version"), str) else None,
        counts=cast(Optional[Dict[str, int]], counts),
        warnings=cast(Optional[list[Dict[str, Any]]], warnings),
        notes=cast(Optional[Dict[str, Any]], notes),
        raw=dict(payload),
        source_path=source_path,
        error=None,
    )


def read_eval_run_summary(path: Pathish) -> EvalRunSummarySnapshot:
    p = Path(path).resolve()
    try:
        payload = read_json_internal(EVAL_RUN_SUMMARY_SCHEMA, p)
        return parse_eval_run_summary(payload, source_path=str(p))
    except Exception as e:
        # Fail-closed snapshot: callers decide how to treat missing/invalid summary.
        return EvalRunSummarySnapshot(
            ok=False,
            schema_version="",
            generated_at="",
            task="",
            run_id="",
            run_dir="",
            passed=False,
            metrics={},
            model_id=None,
            schema_id=None,
            thresholds_profile=None,
            thresholds_version=None,
            counts=None,
            warnings=None,
            notes=None,
            raw={},
            source_path=str(p),
            error=f"eval_run_summary_parse_error: {type(e).__name__}: {e}",
        )


def write_eval_run_summary(path: Pathish, payload: Dict[str, Any]) -> Path:
    return atomic_write_json_internal(EVAL_RUN_SUMMARY_SCHEMA, path, payload)