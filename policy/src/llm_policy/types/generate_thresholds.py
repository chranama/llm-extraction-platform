# policy/src/llm_policy/types/generate_thresholds.py
from __future__ import annotations

from typing import Dict, Optional

from pydantic import BaseModel, Field, field_validator


class GenerateErrorRateThreshold(BaseModel):
    threshold: float = Field(default=0.02, ge=0.0, description="Clamp if error_rate >= threshold.")
    cap: int = Field(default=128, ge=1, description="Token cap applied when error rate is too high.")


class GenerateLatencyP95Threshold(BaseModel):
    """
    steps: threshold_ms -> cap
    Evaluated in descending threshold order.
    """
    steps: Dict[int, int] = Field(default_factory=dict)

    @field_validator("steps")
    @classmethod
    def _validate_steps(cls, v: Dict[int, int]) -> Dict[int, int]:
        for k, cap in v.items():
            if int(k) <= 0:
                raise ValueError("latency_p95_ms.steps keys must be positive integers (ms)")
            if int(cap) <= 0:
                raise ValueError("latency_p95_ms.steps values must be positive integers (cap)")
        return v


class GenerateThresholds(BaseModel):
    """
    Threshold profile for SLO-driven /v1/generate clamping.
    Matches policies/generate_slo_clamp.py semantics.
    """
    min_requests: int = Field(default=10, ge=0)
    safe_cap_on_invalid_snapshot: int = Field(default=128, ge=1)

    error_rate: GenerateErrorRateThreshold = Field(default_factory=GenerateErrorRateThreshold)
    latency_p95_ms: GenerateLatencyP95Threshold = Field(default_factory=GenerateLatencyP95Threshold)