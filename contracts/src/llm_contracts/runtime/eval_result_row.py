# contracts/src/llm_contracts/runtime/eval_result_row.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union, cast

from llm_contracts.schema import read_json_internal, validate_internal

Pathish = Union[str, Path]

EVAL_RESULT_ROW_SCHEMA = "eval_result_row_v1.schema.json"


@dataclass(frozen=True)
class EvalResultRow:
    """
    One row from results.jsonl.

    This is intentionally "thin":
      - stable keys for downstream consumers
      - raw payload retained for forward compatibility

    If you evolve the JSONL schema later, add a new schema file and parser
    instead of breaking this contract.
    """
    ok: bool
    schema_version: str

    # Core identity
    task: str
    run_id: str

    # Row identity / input
    example_id: Optional[str]
    schema_id: Optional[str]

    # Outcome
    passed: Optional[bool]
    score: Optional[float]

    # Free-form diagnostic fields
    error: Optional[str]
    meta: Dict[str, Any]

    raw: Dict[str, Any]
    source_path: Optional[str] = None
    line_no: Optional[int] = None


def _opt_str(payload: Dict[str, Any], key: str) -> Optional[str]:
    v = payload.get(key)
    return v if isinstance(v, str) and v.strip() else None


def _opt_bool(payload: Dict[str, Any], key: str) -> Optional[bool]:
    v = payload.get(key)
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    return None


def _opt_float(payload: Dict[str, Any], key: str) -> Optional[float]:
    v = payload.get(key)
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    return None


def parse_eval_result_row(payload: Dict[str, Any], *, source_path: Optional[str] = None, line_no: Optional[int] = None) -> EvalResultRow:
    """
    Validate + parse a single eval result row (v1).

    Contract:
      - validates against schemas/internal/eval_result_row_v1.schema.json
      - returns a stable dataclass
    """
    validate_internal(EVAL_RESULT_ROW_SCHEMA, payload)

    schema_version = str(payload.get("schema_version", "")).strip()
    if schema_version != "eval_result_row_v1":
        raise ValueError(f"Unsupported eval result row schema_version: {schema_version}")

    task = str(payload["task"]).strip()
    run_id = str(payload["run_id"]).strip()

    # These may or may not exist depending on task; keep optional.
    example_id = _opt_str(payload, "example_id")
    schema_id = _opt_str(payload, "schema_id")

    passed = _opt_bool(payload, "passed")
    score = _opt_float(payload, "score")

    error = _opt_str(payload, "error")
    meta = payload.get("meta")
    meta_dict = dict(meta) if isinstance(meta, dict) else {}

    return EvalResultRow(
        ok=True,
        schema_version=schema_version,
        task=task,
        run_id=run_id,
        example_id=example_id,
        schema_id=schema_id,
        passed=passed,
        score=score,
        error=error,
        meta=meta_dict,
        raw=dict(payload),
        source_path=source_path,
        line_no=line_no,
    )


def read_eval_result_row_json(path: Pathish) -> EvalResultRow:
    """
    Read a single JSON file containing exactly one EvalResultRow payload.
    (Mostly useful for debugging; the canonical store for rows is JSONL.)
    """
    p = Path(path).resolve()
    try:
        payload = read_json_internal(EVAL_RESULT_ROW_SCHEMA, p)
        return parse_eval_result_row(cast(Dict[str, Any], payload), source_path=str(p), line_no=None)
    except Exception as e:
        return EvalResultRow(
            ok=False,
            schema_version="",
            task="",
            run_id="",
            example_id=None,
            schema_id=None,
            passed=None,
            score=None,
            error=f"eval_result_row_parse_error: {type(e).__name__}: {e}",
            meta={},
            raw={},
            source_path=str(p),
            line_no=None,
        )


@dataclass(frozen=True)
class ReadResultsStats:
    """
    Simple counters for JSONL reads.

    - total_lines: total non-empty lines encountered
    - ok_rows: rows successfully parsed + validated
    - skipped_invalid: invalid rows skipped (strict=False)
    - parse_errors: JSON decode errors or non-dict lines
    """
    total_lines: int
    ok_rows: int
    skipped_invalid: int
    parse_errors: int


def _iter_jsonl_lines(path: Path) -> Iterable[tuple[int, str]]:
    """
    Yield (line_no, line_text) for non-empty lines.
    Line numbers are 1-based.
    """
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            yield i, s


def read_results_jsonl(
    path: Pathish,
    *,
    strict: bool = False,
    max_rows: Optional[int] = None,
) -> tuple[List[EvalResultRow], ReadResultsStats]:
    """
    Read + validate eval results from a JSONL file (results.jsonl).

    This is intentionally a THIN helper:
      - no aggregation
      - no pass/fail inference
      - just JSONL parsing + schema validation + stable dataclass output

    Args:
      strict:
        - True: raise on first invalid row (JSON decode or schema validation)
        - False: skip invalid rows and keep counters
      max_rows:
        - optional cap on returned parsed rows (does not stop counting parse errors
          before reaching cap; once cap is reached, iteration stops)

    Returns:
      (rows, stats)
    """
    p = Path(path).resolve()
    rows: List[EvalResultRow] = []

    total_lines = 0
    ok_rows = 0
    skipped_invalid = 0
    parse_errors = 0

    try:
        for line_no, line in _iter_jsonl_lines(p):
            total_lines += 1

            if max_rows is not None and ok_rows >= int(max_rows):
                break

            try:
                obj = json.loads(line)
            except Exception as e:
                if strict:
                    raise ValueError(f"Invalid JSON on line {line_no}: {e}") from e
                parse_errors += 1
                continue

            if not isinstance(obj, dict):
                if strict:
                    raise ValueError(f"Non-object JSON on line {line_no}: expected object, got {type(obj).__name__}")
                parse_errors += 1
                continue

            try:
                row = parse_eval_result_row(cast(Dict[str, Any], obj), source_path=str(p), line_no=line_no)
                rows.append(row)
                ok_rows += 1
            except Exception as e:
                if strict:
                    raise ValueError(f"Invalid row on line {line_no}: {type(e).__name__}: {e}") from e
                skipped_invalid += 1
                continue

    except FileNotFoundError as e:
        if strict:
            raise
        # fail-soft: return empty rows with 1 parse error
        return [], ReadResultsStats(total_lines=0, ok_rows=0, skipped_invalid=0, parse_errors=1)

    return rows, ReadResultsStats(
        total_lines=total_lines,
        ok_rows=ok_rows,
        skipped_invalid=skipped_invalid,
        parse_errors=parse_errors,
    )