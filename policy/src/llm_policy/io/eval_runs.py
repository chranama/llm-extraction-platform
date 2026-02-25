# policy/src/llm_policy/io/eval_runs.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

from pydantic import ValidationError

from llm_contracts.runtime.eval_run_pointer import read_eval_run_pointer

from llm_policy.types.eval_artifact import (
    ContractIssue,
    EvalArtifact,
    EvalRow,
    EvalSummary,
    IssueSeverity,
)

# -----------------------------
# Canonical eval "latest" pointer
# -----------------------------


def default_eval_latest_pointer_path() -> Path:
    """
    Canonical location for the "latest eval run" pointer.

    Priority:
      1) EVAL_LATEST_PATH (explicit override)
      2) eval_out/latest.json (repo-relative)
    """
    env = os.getenv("EVAL_LATEST_PATH", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return (Path("eval_out") / "latest.json").resolve()


def resolve_eval_run_dir(run_dir: str | Path) -> Path:
    """
    Resolve a run_dir argument into an actual eval run directory.

    Accepted values:
      - explicit path to a run directory containing summary.json
      - "latest" (or "." or "") => follow eval_out/latest.json pointer (llm_contracts)

    Fail-closed behavior:
      - if pointer is missing/invalid => returns a non-existent path, so downstream
        loader emits contract issues and gating will deny.
    """
    s = str(run_dir).strip()
    if s in ("", ".", "latest"):
        ptr_path = default_eval_latest_pointer_path()
        snap = read_eval_run_pointer(ptr_path)
        if not snap.ok or not snap.run_dir:
            return Path("__missing_eval_run_dir__")
        return Path(snap.run_dir)
    return Path(run_dir)


# -----------------------------
# Public API
# -----------------------------


@dataclass(frozen=True)
class LoadResult:
    """
    Convenience wrapper so callers can choose how strict to be.
    - artifact: parsed object (summary always, results optional)
    - issues: contract issues found while loading/parsing
    """
    artifact: EvalArtifact
    issues: list[ContractIssue]


def load_eval_artifact(
    run_dir: str | Path,
    *,
    load_results: bool = True,
    max_results: Optional[int] = None,
    allow_missing_results: bool = True,
) -> EvalArtifact:
    """
    Back-compat helper (your CLI currently imports load_eval_artifact).

    Returns only EvalArtifact (drops issues). If you want issues, use load_eval_run_dir().
    """
    res = load_eval_run_dir(
        run_dir,
        load_results=load_results,
        max_results=max_results,
        allow_missing_results=allow_missing_results,
    )
    return res.artifact


def load_eval_run_dir(
    run_dir: str | Path,
    *,
    load_results: bool = True,
    max_results: Optional[int] = None,
    allow_missing_results: bool = True,
) -> LoadResult:
    """
    Load an llm_eval run directory containing:
      - summary.json (required)
      - results.jsonl (optional, depending on load_results / allow_missing_results)

    Enhancements vs previous version:
      - supports run_dir="latest" (follows eval_out/latest.json pointer via llm_contracts)
      - consumes deployment provenance from summary + rows (v2 contracts)
      - FAIL-CLOSED: missing/malformed deployment provenance => error issues
    """
    run_path = resolve_eval_run_dir(run_dir)
    issues: list[ContractIssue] = []

    summary_path = run_path / "summary.json"
    results_path = run_path / "results.jsonl"

    summary = _load_summary(summary_path, issues=issues)

    rows: list[EvalRow] = []
    if load_results:
        if results_path.exists():
            rows = list(iter_results_jsonl(results_path, issues=issues, max_rows=max_results))
        else:
            if not allow_missing_results:
                issues.append(
                    ContractIssue(
                        severity=IssueSeverity.error,
                        code="missing_results",
                        message="results.jsonl not found",
                        context={"path": str(results_path)},
                    )
                )

    # -----------------------------
    # FAIL-CLOSED: deployment provenance checks
    # -----------------------------
    summary_dk = summary.deployment_key.strip() if isinstance(summary.deployment_key, str) and summary.deployment_key.strip() else None
    summary_dep = summary.deployment if isinstance(summary.deployment, dict) and summary.deployment else None

    if summary_dk is None:
        issues.append(
            ContractIssue(
                severity=IssueSeverity.error,
                code="missing_deployment_key",
                message="summary.deployment_key missing/empty (policy must fail-closed)",
                context={"path": str(summary_path)},
            )
        )

    if summary_dep is None:
        issues.append(
            ContractIssue(
                severity=IssueSeverity.error,
                code="missing_deployment",
                message="summary.deployment missing/empty/not an object (policy must fail-closed)",
                context={"path": str(summary_path)},
            )
        )

    # If results are present, require row-level provenance too.
    if load_results and rows:
        missing_row_dk = 0
        missing_row_dep = 0
        mismatched_dk = 0

        for r in rows:
            row_dk = r.deployment_key.strip() if isinstance(r.deployment_key, str) and r.deployment_key.strip() else None
            row_dep = r.deployment if isinstance(r.deployment, dict) and r.deployment else None

            if row_dk is None:
                missing_row_dk += 1
            if row_dep is None:
                missing_row_dep += 1

            if summary_dk is not None and row_dk is not None and row_dk != summary_dk:
                mismatched_dk += 1

        if missing_row_dk:
            issues.append(
                ContractIssue(
                    severity=IssueSeverity.error,
                    code="missing_row_deployment_key",
                    message="One or more rows missing deployment_key (policy must fail-closed)",
                    context={"missing_rows": missing_row_dk, "n_rows": len(rows)},
                )
            )

        if missing_row_dep:
            issues.append(
                ContractIssue(
                    severity=IssueSeverity.error,
                    code="missing_row_deployment",
                    message="One or more rows missing deployment (policy must fail-closed)",
                    context={"missing_rows": missing_row_dep, "n_rows": len(rows)},
                )
            )

        if mismatched_dk:
            issues.append(
                ContractIssue(
                    severity=IssueSeverity.error,
                    code="deployment_key_mismatch",
                    message="One or more rows have deployment_key that does not match summary.deployment_key (policy must fail-closed)",
                    context={"mismatched_rows": mismatched_dk, "n_rows": len(rows), "summary_deployment_key": summary_dk},
                )
            )

    artifact = EvalArtifact(summary=summary, results=rows or None)

    # Add any issues derived from the parsed summary itself (includes deployment requirements)
    issues.extend(summary.contract_issues())

    return LoadResult(artifact=artifact, issues=_dedupe_issues(issues))


def load_summary_file(path: str | Path) -> LoadResult:
    """
    Load only a summary.json (no results.jsonl).
    """
    p = Path(path)
    issues: list[ContractIssue] = []
    summary = _load_summary(p, issues=issues)
    artifact = EvalArtifact(summary=summary, results=None)
    issues.extend(summary.contract_issues())
    return LoadResult(artifact=artifact, issues=_dedupe_issues(issues))


def iter_results_jsonl(
    path: str | Path,
    *,
    issues: Optional[list[ContractIssue]] = None,
    max_rows: Optional[int] = None,
) -> Iterator[EvalRow]:
    """
    Stream-parse results.jsonl and yield EvalRow objects.

    Behavior:
      - skips empty lines
      - skips malformed JSON lines with a ContractIssue (if issues list provided)
      - skips validation failures with a ContractIssue (if issues list provided)
    """
    p = Path(path)
    if not p.exists():
        if issues is not None:
            issues.append(
                ContractIssue(
                    severity=IssueSeverity.error,
                    code="missing_results",
                    message="results.jsonl not found",
                    context={"path": str(p)},
                )
            )
        return iter(())

    n = 0
    with p.open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue

            obj: Any
            try:
                obj = json.loads(line)
            except Exception as e:
                if issues is not None:
                    issues.append(
                        ContractIssue(
                            severity=IssueSeverity.warn,
                            code="results_json_parse_error",
                            message="Failed to parse JSON line in results.jsonl",
                            context={
                                "path": str(p),
                                "lineno": lineno,
                                "error": f"{type(e).__name__}: {e}",
                                "line_preview": line[:300],
                            },
                        )
                    )
                continue

            if not isinstance(obj, dict):
                if issues is not None:
                    issues.append(
                        ContractIssue(
                            severity=IssueSeverity.warn,
                            code="results_row_not_object",
                            message="A results.jsonl line was not a JSON object",
                            context={"path": str(p), "lineno": lineno, "type": type(obj).__name__},
                        )
                    )
                continue

            try:
                row = EvalRow.model_validate(obj)
            except ValidationError as ve:
                if issues is not None:
                    issues.append(
                        ContractIssue(
                            severity=IssueSeverity.warn,
                            code="results_row_validation_error",
                            message="A results.jsonl row failed schema validation",
                            context={
                                "path": str(p),
                                "lineno": lineno,
                                "errors": ve.errors(),
                            },
                        )
                    )
                continue

            yield row
            n += 1
            if max_rows is not None and n >= max_rows:
                break


# -----------------------------
# Internals
# -----------------------------


def _load_summary(path: Path, *, issues: list[ContractIssue]) -> EvalSummary:
    if not path.exists():
        issues.append(
            ContractIssue(
                severity=IssueSeverity.error,
                code="missing_summary",
                message="summary.json not found",
                context={"path": str(path)},
            )
        )
        # Fail-closed: produce a minimal summary that will never pass gating
        return EvalSummary.model_validate({"task": "unknown", "run_id": "unknown", "n_total": 0})

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        issues.append(
            ContractIssue(
                severity=IssueSeverity.error,
                code="summary_json_parse_error",
                message="Failed to parse summary.json",
                context={"path": str(path), "error": f"{type(e).__name__}: {e}"},
            )
        )
        return EvalSummary.model_validate({"task": "unknown", "run_id": "unknown", "n_total": 0})

    if not isinstance(raw, dict):
        issues.append(
            ContractIssue(
                severity=IssueSeverity.error,
                code="summary_not_object",
                message="summary.json root must be a JSON object",
                context={"path": str(path), "type": type(raw).__name__},
            )
        )
        return EvalSummary.model_validate({"task": "unknown", "run_id": "unknown", "n_total": 0})

    try:
        return EvalSummary.model_validate(raw)
    except ValidationError as ve:
        issues.append(
            ContractIssue(
                severity=IssueSeverity.error,
                code="summary_validation_error",
                message="summary.json failed schema validation",
                context={"path": str(path), "errors": ve.errors()},
            )
        )
        # Still try to salvage the minimum keys if present
        minimal = {
            "task": str(raw.get("task") or "unknown"),
            "run_id": str(raw.get("run_id") or "unknown"),
            "n_total": int(raw.get("n_total") or 0),
        }
        return EvalSummary.model_validate(minimal)


def _dedupe_issues(issues: Iterable[ContractIssue]) -> list[ContractIssue]:
    """
    Deduplicate issues by (severity, code, message, context_json).
    """
    seen: set[tuple[str, str, str, str]] = set()
    out: list[ContractIssue] = []
    for it in issues:
        ctx = it.context or {}
        try:
            ctx_s = json.dumps(ctx, sort_keys=True, ensure_ascii=False)
        except Exception:
            ctx_s = str(ctx)
        key = (str(it.severity), it.code, it.message, ctx_s)
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out