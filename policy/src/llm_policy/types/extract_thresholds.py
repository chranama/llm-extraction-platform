# policy/src/llm_policy/types/extract_thresholds.py
from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class ExtractMetricThreshold(BaseModel):
    """
    Generic numeric threshold.
    Interpreted by policies (>= min, <= max).
    """

    min: Optional[float] = Field(default=None, description="Minimum acceptable value (>=).")
    max: Optional[float] = Field(default=None, description="Maximum acceptable value (<=).")
    weight: float = Field(default=1.0, description="Optional weighting for scoring (future use).")
    notes: Optional[str] = Field(default=None)


class ExtractThresholds(BaseModel):
    """
    Threshold profile for extract enablement policies.

    - metrics: thresholds keyed by metric name (e.g. schema_validity_rate)
    - params: policy knobs (Phase 2 uses these)

    Phase 2 expected params (recommended keys):
      - min_n_total: int                     (warning-only; low-confidence)
      - min_n_for_point_estimate: int        (switch CI-low vs point estimate)
      - max_http_5xx_rate: float             (percent units)
      - max_timeout_rate: float              (percent units)
      - max_non_200_rate: float              (percent units; optional)
      - min_field_exact_match_rate: {field: float}  (percent units, optional)
    """

    version: Optional[str] = Field(default=None)
    task: Optional[str] = Field(default=None)

    metrics: Dict[str, ExtractMetricThreshold] = Field(default_factory=dict)
    params: Dict[str, Any] = Field(default_factory=dict)