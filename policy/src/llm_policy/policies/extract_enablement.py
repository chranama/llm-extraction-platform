# policy/src/llm_policy/policies/extract_enablement.py
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from llm_policy.types.decision import Decision, DecisionReason, DecisionStatus, DecisionWarning
from llm_policy.types.eval_artifact import EvalArtifact
from llm_policy.types.extract_thresholds import ExtractThresholds


def _reason(code: str, message: str, context: Optional[dict[str, Any]] = None) -> DecisionReason:
    return DecisionReason(code=code, message=message, context=context or {})


def _warning(code: str, message: str, context: Optional[dict[str, Any]] = None) -> DecisionWarning:
    return DecisionWarning(code=code, message=message, context=context or {})


def _get_metric_threshold(
    thresholds: ExtractThresholds, metric: str
) -> tuple[Optional[float], Optional[float]]:
    mt = thresholds.metrics.get(metric)
    if mt is None:
        return None, None
    return (mt.min if mt.min is not None else None), (mt.max if mt.max is not None else None)


def _get_param(thresholds: ExtractThresholds, key: str, default: Any) -> Any:
    try:
        if isinstance(thresholds.params, dict) and key in thresholds.params:
            return thresholds.params.get(key)
    except Exception:
        pass
    return default


def _as_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def _issue_severity_str(issue: Any) -> str:
    """
    ContractIssue.severity is an Enum in types/eval_artifact.py.
    Normalize it to "error" / "warn" / "info" for robust comparisons.
    """
    sev = getattr(issue, "severity", None)
    if sev is None:
        return "info"
    # Enum -> value
    if hasattr(sev, "value"):
        try:
            return str(sev.value)
        except Exception:
            pass
    return str(sev)


def _issue_context(issue: Any) -> dict[str, Any]:
    """
    ContractIssue may use `.context` OR `.extra` depending on version.
    Be tolerant.
    """
    extra = getattr(issue, "extra", None)
    if isinstance(extra, dict):
        return dict(extra)
    ctx = getattr(issue, "context", None)
    if isinstance(ctx, dict):
        return dict(ctx)
    # If your ContractIssue supports merged_context(), prefer it (future-proof)
    mc = getattr(issue, "merged_context", None)
    if callable(mc):
        try:
            out = mc()
            if isinstance(out, dict):
                return dict(out)
        except Exception:
            pass
    return {}


def _choose_metric_for_gating(
    *,
    value_pct: Optional[float],
    ci95_low_pct: Optional[float],
    n_total: int,
    min_n_for_point_estimate: int,
) -> Tuple[Optional[float], str]:
    """
    Phase 2.2:
      - If n_total < min_n_for_point_estimate and ci95_low exists: use ci95_low
      - Else: use point estimate
    Returns (chosen_value, source) where source in {"ci95_low","point","missing"}.
    """
    if value_pct is None and ci95_low_pct is None:
        return None, "missing"

    if n_total < int(min_n_for_point_estimate or 0) and ci95_low_pct is not None:
        return float(ci95_low_pct), "ci95_low"

    if value_pct is not None:
        return float(value_pct), "point"

    # Fallback: point missing but CI present
    if ci95_low_pct is not None:
        return float(ci95_low_pct), "ci95_low"

    return None, "missing"


def _metric_ci95_low(s: Any, metric: str) -> Optional[float]:
    # Preferred convention: <metric>_ci95_low
    v = _as_float(getattr(s, f"{metric}_ci95_low", None))
    if v is not None:
        return v

    # Back-compat / current EvalSummary field names
    legacy = {
        "schema_validity_rate": "schema_validity_ci95_low",
        "required_present_rate": "required_present_ci95_low",
        "doc_required_exact_match_rate": "doc_required_exact_match_ci95_low",
    }
    alt = legacy.get(metric)
    if not alt:
        return None
    return _as_float(getattr(s, alt, None))


def decide_extract_enablement(
    artifact: EvalArtifact,
    *,
    thresholds: ExtractThresholds,
    thresholds_profile: Optional[str] = None,
) -> Decision:
    """
    Extract enablement policy (pure).

    Phase 2:
      2.1 System unhealthy => inconclusive deny (fail-closed)
      2.2 CI-low gating only when sample is small

    Units:
      - All rates are percent units: 0..100
    """
    s = artifact.summary

    # -----------------------------
    # Policy knobs (params)
    # -----------------------------
    min_n_total = int(_get_param(thresholds, "min_n_total", 0) or 0)
    min_n_for_point_estimate = int(_get_param(thresholds, "min_n_for_point_estimate", 200) or 200)

    max_http_5xx_rate = _as_float(_get_param(thresholds, "max_http_5xx_rate", None))
    max_timeout_rate = _as_float(_get_param(thresholds, "max_timeout_rate", None))
    max_non_200_rate = _as_float(_get_param(thresholds, "max_non_200_rate", None))

    mfem_any = _get_param(thresholds, "min_field_exact_match_rate", {}) or {}
    min_field_exact_match_rate: dict[str, float] = {}
    if isinstance(mfem_any, dict):
        for k, v in mfem_any.items():
            fv = _as_float(v)
            if fv is None:
                continue
            min_field_exact_match_rate[str(k)] = float(fv)

    # -----------------------------
    # Begin decision
    # -----------------------------
    n_total = int(getattr(s, "n_total", 0) or 0)

    reasons: list[DecisionReason] = []
    warnings: list[DecisionWarning] = []
    metrics: Dict[str, Any] = {}

    # Contract issues -> fail-closed deny (but keep as reasons/warnings, not crashes)
    for iss in artifact.contract_issues():
        sev = _issue_severity_str(iss)
        iss_code = str(getattr(iss, "code", "") or "")
        ctx = {"code": iss_code, **_issue_context(iss)}
        msg = str(getattr(iss, "message", "") or "")
        if not msg:
            msg = "Eval artifact contract issue"

        # Specialize deployment provenance failures (cleaner dashboards/logs)
        is_deploy_issue = (
            iss_code.startswith("missing_deployment")
            or iss_code.startswith("missing_row_deployment")
            or iss_code == "deployment_key_mismatch"
        )

        if sev == "error":
            reasons.append(
                _reason(
                    "deployment_contract_error" if is_deploy_issue else "artifact_contract_error",
                    msg,
                    ctx,
                )
            )
        elif sev == "warn":
            warnings.append(_warning("artifact_contract_warn", msg, ctx))
        else:
            warnings.append(_warning("artifact_contract_info", msg, ctx))

    # Sample size warning (non-blocking)
    if min_n_total > 0 and n_total < min_n_total:
        warnings.append(
            _warning(
                "insufficient_sample_size",
                f"n_total={n_total} below min_n_total={min_n_total}; decision is low-confidence",
                {"n_total": n_total, "min_n_total": min_n_total},
            )
        )

    # Always include core provenance in metrics for explainability
    metrics.update(
        {
            "n_total": int(n_total),
            "n_ok": int(getattr(s, "n_ok", 0) or 0),
            "task": str(getattr(s, "task", "") or ""),
            "run_id": str(getattr(s, "run_id", "") or ""),
            "run_dir": str(getattr(s, "run_dir", "") or ""),
            # NEW: deployment provenance (helpful for debugging denials)
            "deployment_key": str(getattr(s, "deployment_key", "") or ""),
            "deployment": getattr(s, "deployment", None),
        }
    )

    # -----------------------------
    # Phase 2.1: System health gates (fail-closed)
    # -----------------------------
    http_5xx_rate = _as_float(getattr(s, "http_5xx_rate", None))
    timeout_rate = _as_float(getattr(s, "timeout_rate", None))
    non_200_rate = _as_float(getattr(s, "non_200_rate", None))

    metrics["http_5xx_rate"] = http_5xx_rate
    metrics["timeout_rate"] = timeout_rate
    metrics["non_200_rate"] = non_200_rate

    # Missing system metrics is fail-closed when the corresponding budget exists
    if max_http_5xx_rate is not None and http_5xx_rate is None:
        reasons.append(_reason("missing_metric", "http_5xx_rate is missing from summary"))
    if max_timeout_rate is not None and timeout_rate is None:
        reasons.append(_reason("missing_metric", "timeout_rate is missing from summary"))
    if max_non_200_rate is not None and non_200_rate is None:
        reasons.append(_reason("missing_metric", "non_200_rate is missing from summary"))

    # Enforce budgets (all in percent units)
    if (
        max_http_5xx_rate is not None
        and http_5xx_rate is not None
        and http_5xx_rate > max_http_5xx_rate
    ):
        reasons.append(
            _reason(
                "system_unhealthy",
                f"http_5xx_rate={http_5xx_rate:.3f}% exceeds budget max_http_5xx_rate={max_http_5xx_rate:.3f}%",
                {
                    "metric": "http_5xx_rate",
                    "current_pct": http_5xx_rate,
                    "max_pct": max_http_5xx_rate,
                    "counts": getattr(s, "http_5xx_counts", None),
                },
            )
        )

    if (
        max_timeout_rate is not None
        and timeout_rate is not None
        and timeout_rate > max_timeout_rate
    ):
        reasons.append(
            _reason(
                "system_unhealthy",
                f"timeout_rate={timeout_rate:.3f}% exceeds budget max_timeout_rate={max_timeout_rate:.3f}%",
                {
                    "metric": "timeout_rate",
                    "current_pct": timeout_rate,
                    "max_pct": max_timeout_rate,
                    "counts": getattr(s, "timeout_counts", None),
                },
            )
        )

    if (
        max_non_200_rate is not None
        and non_200_rate is not None
        and non_200_rate > max_non_200_rate
    ):
        reasons.append(
            _reason(
                "system_unhealthy",
                f"non_200_rate={non_200_rate:.3f}% exceeds budget max_non_200_rate={max_non_200_rate:.3f}%",
                {
                    "metric": "non_200_rate",
                    "current_pct": non_200_rate,
                    "max_pct": max_non_200_rate,
                    "counts": getattr(s, "non_200_counts", None),
                },
            )
        )

    # -----------------------------
    # Phase 2.2: Quality gates with CI-low switching
    # -----------------------------
    def gate_min(metric: str, reason_code: str) -> None:
        """
        Compare chosen(metric) >= thresholds.metrics.<metric>.min
        where chosen is CI-low if sample small and CI exists, else point estimate.
        """
        min_v, _max_v = _get_metric_threshold(thresholds, metric)
        if min_v is None:
            # Not required by thresholds; still record if present.
            v = _as_float(getattr(s, metric, None))
            if v is not None:
                metrics[metric] = v
            return

        v = _as_float(getattr(s, metric, None))
        ci_low = _metric_ci95_low(s, metric)

        chosen, source = _choose_metric_for_gating(
            value_pct=v,
            ci95_low_pct=ci_low,
            n_total=n_total,
            min_n_for_point_estimate=min_n_for_point_estimate,
        )

        # record for explainability
        metrics[metric] = v
        if ci_low is not None:
            metrics[f"{metric}_ci95_low"] = ci_low
        metrics[f"{metric}__gate_source"] = source
        metrics[f"{metric}__gate_value"] = chosen

        if chosen is None:
            reasons.append(
                _reason(
                    "missing_metric", f"{metric} is missing from summary (and no ci95_low present)"
                )
            )
            return

        if float(chosen) < float(min_v):
            reasons.append(
                _reason(
                    reason_code,
                    f"{metric}({source})={float(chosen):.3f}% < min={float(min_v):.3f}%",
                    {
                        "metric": metric,
                        "source": source,
                        "current_pct": float(chosen),
                        "min_pct": float(min_v),
                        "point_pct": v,
                        "ci95_low_pct": ci_low,
                        "n_total": n_total,
                        "min_n_for_point_estimate": min_n_for_point_estimate,
                    },
                )
            )

    gate_min("schema_validity_rate", "schema_validity_too_low")
    gate_min("required_present_rate", "required_present_too_low")
    gate_min("doc_required_exact_match_rate", "doc_required_em_too_low")

    # ---- Per-field EM (params, percent units) ----
    fem = getattr(s, "field_exact_match_rate", None) or {}
    if isinstance(fem, dict):
        metrics["field_exact_match_rate"] = fem
        for field, minv in min_field_exact_match_rate.items():
            cur = fem.get(field)
            if cur is None:
                reasons.append(
                    _reason(
                        "missing_metric",
                        f"field_exact_match_rate.{field} missing",
                        {"field": field},
                    )
                )
                continue

            cur_f = _as_float(cur)
            if cur_f is None:
                reasons.append(
                    _reason(
                        "metric_parse_error",
                        f"field_exact_match_rate.{field} not numeric",
                        {"field": field, "value": cur},
                    )
                )
                continue

            if cur_f < float(minv):
                reasons.append(
                    _reason(
                        "field_em_too_low",
                        f"{field}: {cur_f:.3f}% < min={float(minv):.3f}%",
                        {"field": field, "current_pct": cur_f, "min_pct": float(minv)},
                    )
                )
    else:
        metrics["field_exact_match_rate"] = {}

    # ---- Latency (max thresholds; CI not needed) ----
    for metric in ("latency_p95_ms", "latency_p99_ms"):
        _min_v, max_v = _get_metric_threshold(thresholds, metric)
        v = _as_float(getattr(s, metric, None))
        if v is not None:
            metrics[metric] = float(v)

        if max_v is not None:
            if v is None:
                reasons.append(_reason("missing_metric", f"{metric} is missing from summary"))
            elif float(v) > float(max_v):
                reasons.append(
                    _reason(
                        f"{metric}_too_high",
                        f"{metric}={float(v):.1f}ms > max={float(max_v):.1f}ms",
                        {"metric": metric, "current_ms": float(v), "max_ms": float(max_v)},
                    )
                )

    # -----------------------------
    # Finalize decision
    # -----------------------------
    enable = len(reasons) == 0
    status = DecisionStatus.allow if enable else DecisionStatus.deny

    return Decision(
        policy="extract_enablement",
        pipeline="extract_only",
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
