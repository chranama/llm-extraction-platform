# policy/src/llm_policy/policies/generate_slo_clamp.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from llm_contracts.runtime.generate_slo import GenerateSLOSnapshot
from llm_policy.types.decision import Decision, DecisionReason, DecisionWarning


@dataclass(frozen=True)
class GenerateSloClampThresholds:
    """
    Threshold profile for SLO-driven generate clamping.

    This policy is SHAPING ONLY:
      - It may apply generate_max_new_tokens_cap
      - It must never gate extract
      - It must never flip Decision.status

    Option A behavior (your choice):
      - If snapshot is missing/unreadable/invalid => NO CLAMP applied (but warning + metrics)
      - If insufficient traffic => NO CLAMP applied (reason + metrics)
    """

    # Minimum traffic before applying clamp logic
    min_requests: int = 10

    # Safe clamp (kept for compatibility; not used when Option A is enabled)
    safe_cap_on_invalid_snapshot: int = 128

    # Error-rate clamp
    error_rate_threshold: float = 0.02
    error_rate_cap: int = 128

    # latency_p95_ms threshold(ms) -> cap (evaluated DESC)
    latency_p95_steps: Dict[int, int] | None = None


def thresholds_from_mapping(m: Mapping[str, Any]) -> GenerateSloClampThresholds:
    """
    Convert parsed YAML mapping into thresholds object.

    Kept local so policy remains self-contained and testable.
    """
    min_requests = int(m.get("min_requests") or 10)
    safe_cap = int(m.get("safe_cap_on_invalid_snapshot") or 128)

    er = m.get("error_rate") or {}
    error_rate_threshold = float(er.get("threshold") or 0.02)
    error_rate_cap = int(er.get("cap") or 128)

    lat = m.get("latency_p95_ms") or {}
    steps_raw = lat.get("steps") or {}
    steps: Dict[int, int] = {}
    for k, v in dict(steps_raw).items():
        try:
            steps[int(k)] = int(v)
        except Exception:
            continue

    return GenerateSloClampThresholds(
        min_requests=min_requests,
        safe_cap_on_invalid_snapshot=safe_cap,
        error_rate_threshold=error_rate_threshold,
        error_rate_cap=error_rate_cap,
        latency_p95_steps=steps or None,
    )


def _as_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _as_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def decide_generate_slo_clamp(
    *,
    base: Decision,
    slo: GenerateSLOSnapshot | None,
    thresholds: GenerateSloClampThresholds,
    policy_name: str = "generate_slo_clamp",
    enabled: bool = True,
) -> Decision:
    """
    Enrich an existing Decision with generate clamp derived from SLO telemetry.

    Invariants:
      - NEVER change base.status
      - NEVER change base.enable_extract
      - ONLY shape generate_max_new_tokens_cap
      - Always return a new Decision (copy/update)

    Option A (no clamp on missing snapshot):
      - If enabled and snapshot is missing/invalid => NO CLAMP, but emit warning + metrics.
    """
    if not enabled:
        return base

    cap: Optional[int] = None
    reasons = list(base.reasons)
    warnings = list(base.warnings)
    metrics = dict(base.metrics)

    # ------------------------------------------------------------
    # Missing snapshot => no clamp (Option A)
    # ------------------------------------------------------------
    if slo is None:
        warnings.append(
            DecisionWarning(
                code="generate_slo_snapshot_missing",
                message="Generate SLO snapshot missing; no clamp applied.",
                context={"policy": policy_name},
            )
        )
        reasons.append(
            DecisionReason(
                code="generate_slo_no_snapshot",
                message="No generate SLO snapshot available; skipping clamp.",
                context={},
            )
        )
        return base.model_copy(
            update={
                "generate_max_new_tokens_cap": None,
                "reasons": reasons,
                "warnings": warnings,
                "metrics": metrics,
            }
        )

    # Attach SLO metrics for visibility (robust to partial/malformed objects)
    total_requests = _as_int(getattr(slo, "total_requests", None))
    error_rate = _as_float(getattr(slo, "error_rate", None))
    latency_p95_ms = _as_float(getattr(slo, "latency_p95_ms", None))
    completion_tokens_p95 = _as_float(getattr(slo, "completion_tokens_p95", None))
    source_path = getattr(slo, "source_path", None)
    slo_error = getattr(slo, "error", None)

    metrics.update(
        {
            "generate_slo_total_requests": total_requests,
            "generate_slo_error_rate": error_rate,
            "generate_slo_latency_p95_ms": latency_p95_ms,
            "generate_slo_completion_tokens_p95": completion_tokens_p95,
            "generate_slo_source_path": source_path,
            "generate_slo_error": slo_error,
        }
    )

    # ------------------------------------------------------------
    # Invalid snapshot => no clamp (Option A)
    # ------------------------------------------------------------
    if slo_error:
        warnings.append(
            DecisionWarning(
                code="generate_slo_snapshot_invalid",
                message="Generate SLO snapshot invalid/unreadable; no clamp applied.",
                context={
                    "policy": policy_name,
                    "source_path": source_path,
                    "error": slo_error,
                },
            )
        )
        reasons.append(
            DecisionReason(
                code="generate_slo_invalid_snapshot_no_clamp",
                message="Skipping generate clamp due to invalid SLO snapshot (Option A).",
                context={},
            )
        )
        return base.model_copy(
            update={
                "generate_max_new_tokens_cap": None,
                "reasons": reasons,
                "warnings": warnings,
                "metrics": metrics,
            }
        )

    # If required numeric fields are missing, treat as invalid => no clamp (Option A)
    if total_requests is None or error_rate is None or latency_p95_ms is None:
        warnings.append(
            DecisionWarning(
                code="generate_slo_snapshot_incomplete",
                message="Generate SLO snapshot missing required fields; no clamp applied.",
                context={
                    "policy": policy_name,
                    "source_path": source_path,
                    "total_requests": total_requests,
                    "error_rate": error_rate,
                    "latency_p95_ms": latency_p95_ms,
                },
            )
        )
        reasons.append(
            DecisionReason(
                code="generate_slo_incomplete_snapshot_no_clamp",
                message="Skipping generate clamp due to incomplete SLO snapshot (Option A).",
                context={},
            )
        )
        return base.model_copy(
            update={
                "generate_max_new_tokens_cap": None,
                "reasons": reasons,
                "warnings": warnings,
                "metrics": metrics,
            }
        )

    # ------------------------------------------------------------
    # Low traffic => no clamp (avoid noise / flapping)
    # ------------------------------------------------------------
    if total_requests < int(thresholds.min_requests):
        reasons.append(
            DecisionReason(
                code="generate_slo_insufficient_traffic",
                message="Insufficient traffic to apply generate clamp.",
                context={
                    "total_requests": total_requests,
                    "min_requests": thresholds.min_requests,
                },
            )
        )
        return base.model_copy(
            update={
                "generate_max_new_tokens_cap": None,
                "reasons": reasons,
                "warnings": warnings,
                "metrics": metrics,
            }
        )

    # ------------------------------------------------------------
    # Error-rate clamp (highest priority)
    # ------------------------------------------------------------
    if error_rate >= float(thresholds.error_rate_threshold):
        cap = int(thresholds.error_rate_cap)
        reasons.append(
            DecisionReason(
                code="generate_slo_error_rate_high",
                message="Generate clamp applied due to elevated error rate.",
                context={
                    "error_rate": error_rate,
                    "threshold": thresholds.error_rate_threshold,
                    "cap": cap,
                },
            )
        )
        return base.model_copy(
            update={
                "generate_max_new_tokens_cap": cap,
                "reasons": reasons,
                "warnings": warnings,
                "metrics": metrics,
            }
        )

    # ------------------------------------------------------------
    # Latency-based clamp (stepwise)
    # ------------------------------------------------------------
    steps = thresholds.latency_p95_steps or {}
    for thr_ms in sorted(steps.keys(), reverse=True):
        if latency_p95_ms >= float(thr_ms):
            cap = int(steps[thr_ms])
            reasons.append(
                DecisionReason(
                    code="generate_slo_latency_high",
                    message="Generate clamp applied due to elevated p95 latency.",
                    context={
                        "p95_ms": latency_p95_ms,
                        "threshold_ms": thr_ms,
                        "cap": cap,
                    },
                )
            )
            return base.model_copy(
                update={
                    "generate_max_new_tokens_cap": cap,
                    "reasons": reasons,
                    "warnings": warnings,
                    "metrics": metrics,
                }
            )

    # ------------------------------------------------------------
    # No clamp needed
    # ------------------------------------------------------------
    reasons.append(
        DecisionReason(
            code="generate_slo_no_clamp",
            message="Generate SLO within thresholds; no clamp applied.",
            context={
                "error_rate": error_rate,
                "latency_p95_ms": latency_p95_ms,
            },
        )
    )
    return base.model_copy(
        update={
            "generate_max_new_tokens_cap": None,
            "reasons": reasons,
            "warnings": warnings,
            "metrics": metrics,
        }
    )