# policy/src/llm_policy/policies/extract_enablement.py
from __future__ import annotations

from typing import Any, Dict, Optional

from llm_policy.types.decision import (
    Decision,
    DecisionReason,
    DecisionStatus,
    DecisionWarning,
)
from llm_policy.types.eval_artifact import EvalArtifact
from llm_policy.types.extract_thresholds import ExtractThresholds


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


def _get_metric_threshold(thresholds: ExtractThresholds, metric: str) -> tuple[Optional[float], Optional[float]]:
    """
    Return (min, max) for a metric from thresholds.metrics if present.
    """
    mt = thresholds.metrics.get(metric)
    if mt is None:
        return None, None
    min_v = mt.min if mt.min is not None else None
    max_v = mt.max if mt.max is not None else None
    return min_v, max_v


def _get_param(thresholds: ExtractThresholds, key: str, default: Any) -> Any:
    """
    Read a policy knob from thresholds.params with a default.
    """
    try:
        if isinstance(thresholds.params, dict) and key in thresholds.params:
            return thresholds.params.get(key)
    except Exception:
        pass
    return default


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
      - Missing required metrics => deny
    """

    s = artifact.summary

    # -----------------------------
    # Policy knobs (params)
    # -----------------------------
    min_n_total = int(_get_param(thresholds, "min_n_total", 0) or 0)

    # NOTE: `min_field_exact_match_rate` is stored under params as a mapping:
    #   params.min_field_exact_match_rate: { field_name: 0.8, ... }
    mfem_any = _get_param(thresholds, "min_field_exact_match_rate", {}) or {}
    min_field_exact_match_rate: dict[str, float] = {}
    if isinstance(mfem_any, dict):
        for k, v in mfem_any.items():
            try:
                if v is None:
                    continue
                min_field_exact_match_rate[str(k)] = float(v)
            except Exception:
                continue

    # -----------------------------
    # Begin decision
    # -----------------------------
    n_total = int(getattr(s, "n_total", 0) or 0)

    reasons: list[DecisionReason] = []
    warnings: list[DecisionWarning] = []
    metrics: Dict[str, Any] = {}

    # Sample size warning (non-blocking)
    if min_n_total > 0 and n_total < min_n_total:
        warnings.append(
            _warning(
                "insufficient_sample_size",
                f"n_total={n_total} below min_n_total={min_n_total}; decision is low-confidence",
                {"n_total": n_total, "min_n_total": min_n_total},
            )
        )

    # ---- Schema validity ----
    sv = getattr(s, "schema_validity_rate", None)
    min_sv, _max_sv = _get_metric_threshold(thresholds, "schema_validity_rate")
    if sv is None:
        reasons.append(_reason("missing_metric", "schema_validity_rate is missing from summary"))
    else:
        metrics["schema_validity_rate"] = float(sv)
        if min_sv is not None and float(sv) < float(min_sv):
            reasons.append(
                _reason(
                    "schema_validity_too_low",
                    f"{float(sv):.3f} < min_schema_validity_rate={float(min_sv):.3f}",
                    {"current": float(sv), "min": float(min_sv)},
                )
            )

    # ---- Required present rate ----
    rp = getattr(s, "required_present_rate", None)
    min_rp, _max_rp = _get_metric_threshold(thresholds, "required_present_rate")
    if min_rp is not None:
        if rp is None:
            reasons.append(_reason("missing_metric", "required_present_rate is missing from summary"))
        else:
            metrics["required_present_rate"] = float(rp)
            if float(rp) < float(min_rp):
                reasons.append(
                    _reason(
                        "required_present_too_low",
                        f"{float(rp):.3f} < min_required_present_rate={float(min_rp):.3f}",
                        {"current": float(rp), "min": float(min_rp)},
                    )
                )
    else:
        # still record if present
        if rp is not None:
            metrics["required_present_rate"] = float(rp)

    # ---- Doc EM ----
    em = getattr(s, "doc_required_exact_match_rate", None)
    min_em, _max_em = _get_metric_threshold(thresholds, "doc_required_exact_match_rate")
    if min_em is not None:
        if em is None:
            reasons.append(_reason("missing_metric", "doc_required_exact_match_rate missing from summary"))
        else:
            metrics["doc_required_exact_match_rate"] = float(em)
            if float(em) < float(min_em):
                reasons.append(
                    _reason(
                        "doc_required_em_too_low",
                        f"{float(em):.3f} < min_doc_required_exact_match_rate={float(min_em):.3f}",
                        {"current": float(em), "min": float(min_em)},
                    )
                )
    else:
        if em is not None:
            metrics["doc_required_exact_match_rate"] = float(em)

    # ---- Per-field EM (params) ----
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
            else:
                try:
                    cur_f = float(cur)
                    if cur_f < float(minv):
                        reasons.append(
                            _reason(
                                "field_em_too_low",
                                f"{field}: {cur_f:.3f} < min={float(minv):.3f}",
                                {"field": field, "current": cur_f, "min": float(minv)},
                            )
                        )
                except Exception:
                    reasons.append(
                        _reason(
                            "metric_parse_error",
                            f"field_exact_match_rate.{field} not numeric",
                            {"field": field, "value": cur},
                        )
                    )
    else:
        metrics["field_exact_match_rate"] = {}

    # ---- Latency ----
    lat_p95 = getattr(s, "latency_p95_ms", None)
    _min_p95, max_p95 = _get_metric_threshold(thresholds, "latency_p95_ms")
    if max_p95 is not None and lat_p95 is not None:
        metrics["latency_p95_ms"] = float(lat_p95)
        if float(lat_p95) > float(max_p95):
            reasons.append(
                _reason(
                    "latency_p95_too_high",
                    f"{float(lat_p95):.1f}ms > max={float(max_p95):.1f}ms",
                    {"current_ms": float(lat_p95), "max_ms": float(max_p95)},
                )
            )
    else:
        if lat_p95 is not None:
            metrics["latency_p95_ms"] = float(lat_p95)

    lat_p99 = getattr(s, "latency_p99_ms", None)
    _min_p99, max_p99 = _get_metric_threshold(thresholds, "latency_p99_ms")
    if max_p99 is not None and lat_p99 is not None:
        metrics["latency_p99_ms"] = float(lat_p99)
        if float(lat_p99) > float(max_p99):
            reasons.append(
                _reason(
                    "latency_p99_too_high",
                    f"{float(lat_p99):.1f}ms > max={float(max_p99):.1f}ms",
                    {"current_ms": float(lat_p99), "max_ms": float(max_p99)},
                )
            )
    else:
        if lat_p99 is not None:
            metrics["latency_p99_ms"] = float(lat_p99)

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