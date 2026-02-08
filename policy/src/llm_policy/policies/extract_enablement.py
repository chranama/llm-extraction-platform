# policy/src/llm_policy/policies/extract_enablement.py
from __future__ import annotations

from typing import Any, Dict, Optional

from llm_policy.policies.health_gate import health_gate_from_eval
from llm_policy.types.decision import (
    Decision,
    DecisionReason,
    DecisionStatus,
    DecisionWarning,
)
from llm_policy.types.eval_artifact import EvalArtifact
from llm_policy.types.thresholds import ExtractThresholds


def _reason(code: str, message: str, context: Optional[dict[str, Any]] = None) -> DecisionReason:
    return DecisionReason(code=code, message=message, context=context or {})


def _warning(code: str, message: str, context: Optional[dict[str, Any]] = None) -> DecisionWarning:
    return DecisionWarning(code=code, message=message, context=context or {})


def _coerce_reasons(items: Any) -> list[DecisionReason]:
    if not items or not isinstance(items, list):
        return []

    out: list[DecisionReason] = []
    for it in items:
        if isinstance(it, DecisionReason):
            out.append(it)
        elif isinstance(it, DecisionWarning):
            out.append(_reason(it.code, it.message, dict(it.context or {})))
        elif isinstance(it, dict):
            code = str(it.get("code") or "issue")
            msg = str(it.get("message") or "")
            if not msg:
                continue
            ctx_any = it.get("context") or it.get("extra")
            ctx = ctx_any if isinstance(ctx_any, dict) else {}
            out.append(_reason(code, msg, ctx))
    return out


def _coerce_warnings(items: Any) -> list[DecisionWarning]:
    if not items or not isinstance(items, list):
        return []

    out: list[DecisionWarning] = []
    for it in items:
        if isinstance(it, DecisionWarning):
            out.append(it)
        elif isinstance(it, DecisionReason):
            out.append(_warning(it.code, it.message, dict(it.context or {})))
        elif isinstance(it, dict):
            code = str(it.get("code") or "warning")
            msg = str(it.get("message") or "")
            if not msg:
                continue
            ctx_any = it.get("context") or it.get("extra")
            ctx = ctx_any if isinstance(ctx_any, dict) else {}
            out.append(_warning(code, msg, ctx))
    return out


def decide_extract_enablement(
    artifact: EvalArtifact,
    *,
    thresholds: ExtractThresholds,
    thresholds_profile: Optional[str] = None,
) -> Decision:
    """
    Extract enablement policy (pure).

    Responsibilities:
      - Decide allow/deny for extract
      - Populate enable_extract, status, reasons, warnings, metrics
      - NEVER decide pipeline
      - NEVER assume generate clamp is present

    Fail-closed:
      - Health gate block => deny
      - Missing required metrics => deny
    """

    # ------------------------------------------------------------------
    # 1) Health gate (hard fail)
    # ------------------------------------------------------------------
    hg = health_gate_from_eval(
        artifact,
        thresholds=thresholds,
        thresholds_profile=thresholds_profile,
    )

    s = artifact.summary

    if getattr(hg, "enable_extract", None) is False:
        reasons = _coerce_reasons(getattr(hg, "reasons", None))
        warnings = _coerce_warnings(getattr(hg, "warnings", None))
        metrics = dict(getattr(hg, "metrics", {}) or {})

        if not reasons:
            reasons = [
                _reason(
                    "health_gate_block",
                    "Health gate blocked extraction.",
                )
            ]

        return Decision(
            policy="extract_enablement",
            status=DecisionStatus.deny,
            enable_extract=False,
            thresholds_profile=thresholds_profile,
            reasons=reasons,
            warnings=warnings,
            metrics=metrics,
            eval_task=str(getattr(s, "task", "") or ""),
            eval_run_id=str(getattr(s, "run_id", "") or ""),
            eval_run_dir=str(getattr(s, "run_dir", "") or ""),
        )

    # ------------------------------------------------------------------
    # 2) Quality / latency gating
    # ------------------------------------------------------------------
    n_total = int(getattr(s, "n_total", 0) or 0)

    reasons: list[DecisionReason] = []
    warnings: list[DecisionWarning] = []
    metrics: Dict[str, Any] = {}

    # Carry forward HG warnings/metrics
    warnings.extend(_coerce_warnings(getattr(hg, "warnings", None)))
    metrics.update(dict(getattr(hg, "metrics", {}) or {}))

    # Downgrade HG reasons to warnings on pass
    for r in _coerce_reasons(getattr(hg, "reasons", None)):
        warnings.append(_warning(r.code, r.message, dict(r.context or {})))

    # Sample size warning (non-blocking)
    if n_total < thresholds.min_n_total:
        warnings.append(
            _warning(
                "insufficient_sample_size",
                f"n_total={n_total} below min_n_total={thresholds.min_n_total}; decision is low-confidence",
                {"n_total": n_total, "min_n_total": thresholds.min_n_total},
            )
        )

    # ---- Schema validity ----
    sv = getattr(s, "schema_validity_rate", None)
    if sv is None:
        reasons.append(_reason("missing_metric", "schema_validity_rate is missing from summary"))
    else:
        metrics["schema_validity_rate"] = float(sv)
        if float(sv) < thresholds.min_schema_validity_rate:
            reasons.append(
                _reason(
                    "schema_validity_too_low",
                    f"{float(sv):.3f} < min_schema_validity_rate={thresholds.min_schema_validity_rate:.3f}",
                    {"current": float(sv), "min": float(thresholds.min_schema_validity_rate)},
                )
            )

    # ---- Required present rate ----
    if thresholds.min_required_present_rate is not None:
        rp = getattr(s, "required_present_rate", None)
        if rp is None:
            reasons.append(_reason("missing_metric", "required_present_rate is missing from summary"))
        else:
            metrics["required_present_rate"] = float(rp)
            if float(rp) < float(thresholds.min_required_present_rate):
                reasons.append(
                    _reason(
                        "required_present_too_low",
                        f"{float(rp):.3f} < min_required_present_rate={thresholds.min_required_present_rate:.3f}",
                        {"current": float(rp), "min": float(thresholds.min_required_present_rate)},
                    )
                )

    # ---- Doc EM ----
    if thresholds.min_doc_required_exact_match_rate is not None:
        em = getattr(s, "doc_required_exact_match_rate", None)
        if em is None:
            reasons.append(_reason("missing_metric", "doc_required_exact_match_rate missing from summary"))
        else:
            metrics["doc_required_exact_match_rate"] = float(em)
            if float(em) < float(thresholds.min_doc_required_exact_match_rate):
                reasons.append(
                    _reason(
                        "doc_required_em_too_low",
                        f"{float(em):.3f} < min_doc_required_exact_match_rate={thresholds.min_doc_required_exact_match_rate:.3f}",
                        {"current": float(em), "min": float(thresholds.min_doc_required_exact_match_rate)},
                    )
                )

    # ---- Per-field EM ----
    fem = getattr(s, "field_exact_match_rate", None) or {}
    metrics["field_exact_match_rate"] = fem
    for field, minv in (thresholds.min_field_exact_match_rate or {}).items():
        cur = fem.get(field)
        if cur is None:
            reasons.append(_reason("missing_metric", f"field_exact_match_rate.{field} missing", {"field": field}))
        elif float(cur) < float(minv):
            reasons.append(
                _reason(
                    "field_em_too_low",
                    f"{field}: {float(cur):.3f} < min={float(minv):.3f}",
                    {"field": field, "current": float(cur), "min": float(minv)},
                )
            )

    # ---- Latency ----
    if thresholds.max_latency_p95_ms is not None and getattr(s, "latency_p95_ms", None) is not None:
        metrics["latency_p95_ms"] = float(s.latency_p95_ms)
        if float(s.latency_p95_ms) > float(thresholds.max_latency_p95_ms):
            reasons.append(
                _reason(
                    "latency_p95_too_high",
                    f"{float(s.latency_p95_ms):.1f}ms > max={float(thresholds.max_latency_p95_ms):.1f}ms",
                    {"current_ms": float(s.latency_p95_ms), "max_ms": float(thresholds.max_latency_p95_ms)},
                )
            )

    if thresholds.max_latency_p99_ms is not None and getattr(s, "latency_p99_ms", None) is not None:
        metrics["latency_p99_ms"] = float(s.latency_p99_ms)
        if float(s.latency_p99_ms) > float(thresholds.max_latency_p99_ms):
            reasons.append(
                _reason(
                    "latency_p99_too_high",
                    f"{float(s.latency_p99_ms):.1f}ms > max={float(thresholds.max_latency_p99_ms):.1f}ms",
                    {"current_ms": float(s.latency_p99_ms), "max_ms": float(thresholds.max_latency_p99_ms)},
                )
            )

    metrics.update(
        {
            "n_total": int(getattr(s, "n_total", 0) or 0),
            "n_ok": int(getattr(s, "n_ok", 0) or 0),
        }
    )

    enable = len(reasons) == 0
    status = DecisionStatus.allow if enable else DecisionStatus.deny

    return Decision(
        policy="extract_enablement",
        status=status,
        enable_extract=enable,
        thresholds_profile=thresholds_profile,
        reasons=reasons,
        warnings=warnings,
        metrics=metrics,
        eval_task=str(getattr(s, "task", "") or ""),
        eval_run_id=str(getattr(s, "run_id", "") or ""),
        eval_run_dir=str(getattr(s, "run_dir", "") or ""),
    )