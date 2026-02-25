# policy/src/llm_policy/types/eval_artifact.py
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


# -------------------------
# Contract / issues
# -------------------------


class IssueSeverity(str, Enum):
    error = "error"
    warn = "warn"
    info = "info"


class ContractIssue(BaseModel):
    """
    Structured issues found when validating an eval artifact contract.

    IMPORTANT: io/eval_runs.py uses `context=...` and reads `it.context`.
    Older code may use `extra=...`. We support BOTH.
    """

    model_config = ConfigDict(extra="ignore")

    severity: IssueSeverity
    code: str
    message: str

    # Newer IO uses `context`
    context: Optional[Dict[str, Any]] = None
    # Back-compat: allow `extra`
    extra: Optional[Dict[str, Any]] = None

    def merged_context(self) -> Dict[str, Any]:
        if isinstance(self.context, dict):
            return self.context
        if isinstance(self.extra, dict):
            return self.extra
        return {}


# -------------------------
# Summary (summary.json)
# -------------------------


class EvalSummary(BaseModel):
    """
    Typed view over llm_eval summary.json.

    IMPORTANT:
    - Rates are percent units: 0..100 (matches llm_eval extraction_scoring.py)
    """

    model_config = ConfigDict(extra="ignore")

    artifact_version: Optional[str] = None

    # Identity / provenance
    task: str
    run_id: str
    dataset: Optional[str] = None
    split: Optional[str] = None
    schema_id: Optional[str] = None
    base_url: Optional[str] = None
    model_override: Optional[str] = None
    max_examples: Optional[int] = None
    run_dir: Optional[str] = None

    # -------------------------
    # NEW: deployment provenance (v2 contracts)
    # -------------------------
    deployment_key: Optional[str] = None
    deployment: Optional[Dict[str, Any]] = None

    # Core counts
    n_total: int = Field(default=0, ge=0)
    n_ok: int = Field(default=0, ge=0)

    # -------------------------
    # Quality metrics (percent units)
    # -------------------------
    schema_validity_rate: Optional[float] = None
    required_present_rate: Optional[float] = None
    doc_required_exact_match_rate: Optional[float] = None
    field_exact_match_rate: Optional[Dict[str, float]] = None

    # -------------------------
    # System health metrics (percent units)
    # -------------------------
    non_200_rate: Optional[float] = None
    http_5xx_rate: Optional[float] = None
    timeout_rate: Optional[float] = None

    non_200_counts: Optional[Dict[str, int]] = None
    http_5xx_counts: Optional[Dict[str, int]] = None
    timeout_counts: Optional[Dict[str, int]] = None

    # Repair / cache aggregates (optional)
    n_invalid_initial: Optional[int] = Field(default=None, ge=0)
    n_repair_attempted: Optional[int] = Field(default=None, ge=0)
    n_repair_success: Optional[int] = Field(default=None, ge=0)
    repair_success_rate: Optional[float] = None

    n_cached: Optional[int] = Field(default=None, ge=0)
    cache_hit_rate: Optional[float] = None

    # Error aggregates
    error_code_counts: Optional[Dict[str, int]] = None
    status_code_counts: Optional[Dict[str, int]] = None
    error_stage_counts: Optional[Dict[str, int]] = None

    # Latency
    latency_p50_ms: Optional[float] = None
    latency_p95_ms: Optional[float] = None
    latency_p99_ms: Optional[float] = None

    # -------------------------
    # Phase 1.3: CI lows + gate values (percent units)
    # -------------------------
    schema_validity_ci95_low: Optional[float] = None
    required_present_ci95_low: Optional[float] = None
    doc_required_exact_match_ci95_low: Optional[float] = None

    schema_validity_gate: Optional[float] = None
    required_present_gate: Optional[float] = None
    doc_required_exact_match_gate: Optional[float] = None

    def contract_issues(self) -> List[ContractIssue]:
        issues: List[ContractIssue] = []

        if not isinstance(self.task, str) or not self.task.strip():
            issues.append(
                ContractIssue(severity=IssueSeverity.error, code="missing_task", message="summary.task is missing/empty")
            )
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            issues.append(
                ContractIssue(severity=IssueSeverity.error, code="missing_run_id", message="summary.run_id is missing/empty")
            )

        if self.n_ok > self.n_total:
            issues.append(
                ContractIssue(
                    severity=IssueSeverity.error,
                    code="inconsistent_counts",
                    message="n_ok cannot exceed n_total",
                    context={"n_total": self.n_total, "n_ok": self.n_ok},
                )
            )

        if self.n_total == 0:
            issues.append(
                ContractIssue(
                    severity=IssueSeverity.warn,
                    code="zero_examples",
                    message="n_total == 0; metrics are not meaningful",
                )
            )

        # -------------------------
        # FAIL-CLOSED: deployment provenance MUST be present
        # -------------------------
        dk = self.deployment_key
        if not isinstance(dk, str) or not dk.strip():
            issues.append(
                ContractIssue(
                    severity=IssueSeverity.error,
                    code="missing_deployment_key",
                    message="summary.deployment_key missing/empty (policy must fail-closed)",
                )
            )

        dep = self.deployment
        if not isinstance(dep, dict) or not dep:
            issues.append(
                ContractIssue(
                    severity=IssueSeverity.error,
                    code="missing_deployment",
                    message="summary.deployment missing/empty/not an object (policy must fail-closed)",
                )
            )

        # Validate percent-valued metrics if present
        for name in (
            "schema_validity_rate",
            "required_present_rate",
            "doc_required_exact_match_rate",
            "non_200_rate",
            "http_5xx_rate",
            "timeout_rate",
            "schema_validity_ci95_low",
            "required_present_ci95_low",
            "doc_required_exact_match_ci95_low",
            "schema_validity_gate",
            "required_present_gate",
            "doc_required_exact_match_gate",
        ):
            v = getattr(self, name, None)
            if v is None:
                continue
            try:
                fv = float(v)
            except Exception:
                issues.append(
                    ContractIssue(
                        severity=IssueSeverity.error,
                        code="invalid_metric_type",
                        message=f"{name} must be numeric (percent units)",
                        context={"metric": name, "value": v},
                    )
                )
                continue
            if not (0.0 <= fv <= 100.0):
                issues.append(
                    ContractIssue(
                        severity=IssueSeverity.error,
                        code="invalid_metric_range",
                        message=f"{name} must be in [0,100] (percent units)",
                        context={"metric": name, "value": fv},
                    )
                )

        # Helpful warning: all 500s
        if self.status_code_counts and self.n_total:
            if _count_for_status(self.status_code_counts, "500") == self.n_total:
                issues.append(
                    ContractIssue(
                        severity=IssueSeverity.warn,
                        code="all_500s",
                        message="All requests returned HTTP 500; operational failure, not model quality",
                        context={"status_code_counts": self.status_code_counts},
                    )
                )

        return issues

    def metric_value(self, metric: str) -> Optional[float]:
        v = getattr(self, metric, None)
        if v is None:
            return None
        try:
            return float(v)
        except Exception:
            return None

    def metric_ci95_low(self, metric: str) -> Optional[float]:
        key = f"{metric}_ci95_low"
        v = getattr(self, key, None)
        if v is None:
            return None
        try:
            return float(v)
        except Exception:
            return None


# -------------------------
# Per-example rows (results.jsonl)
# -------------------------


class EvalRow(BaseModel):
    model_config = ConfigDict(extra="ignore")

    doc_id: str
    schema_id: Optional[str] = None

    ok: bool = False
    status_code: Optional[int] = None
    error_code: Optional[str] = None
    error_stage: Optional[str] = None

    latency_ms: Optional[float] = None
    cached: Optional[bool] = None
    repair_attempted: Optional[bool] = None
    model: Optional[str] = None

    expected: Optional[Dict[str, Any]] = None
    predicted: Optional[Dict[str, Any]] = None

    field_correct: Optional[Dict[str, Optional[bool]]] = None
    required_present_non_null: Optional[bool] = None
    required_all_correct: Optional[bool] = None

    extra: Optional[Dict[str, Any]] = None

    # -------------------------
    # NEW: deployment provenance (v2 contracts)
    # -------------------------
    deployment_key: Optional[str] = None
    deployment: Optional[Dict[str, Any]] = None


# -------------------------
# Wrapper
# -------------------------


@dataclass(frozen=True)
class EvalArtifact:
    """
    io/eval_runs.py uses `results=...` so keep this field name as `results`.
    """
    summary: EvalSummary
    results: Optional[List[EvalRow]] = None

    def contract_issues(self) -> List[ContractIssue]:
        issues = self.summary.contract_issues()
        if self.results is not None and self.summary.n_total and len(self.results) != self.summary.n_total:
            issues.append(
                ContractIssue(
                    severity=IssueSeverity.info,
                    code="results_length_mismatch",
                    message="len(results.jsonl) != summary.n_total",
                    context={"len_results": len(self.results), "n_total": self.summary.n_total},
                )
            )
        return issues


# -------------------------
# Small helpers
# -------------------------


def _count_for_status(counts: Dict[str, int], code_str: str) -> int:
    v = counts.get(code_str)
    try:
        return int(v or 0)
    except Exception:
        return 0