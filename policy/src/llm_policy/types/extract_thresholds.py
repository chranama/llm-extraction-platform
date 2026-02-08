# policy/src/llm_policy/types/extract_thresholds.py
from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class ExtractMetricThreshold(BaseModel):
    """
    Generic numeric threshold with a comparison direction.
    """
    min: Optional[float] = Field(default=None, description="Minimum acceptable value (>=).")
    max: Optional[float] = Field(default=None, description="Maximum acceptable value (<=).")
    weight: float = Field(default=1.0, description="Optional weighting for scoring (future use).")
    notes: Optional[str] = Field(default=None)


class ExtractThresholds(BaseModel):
    """
    Threshold profile for extract enablement policies.

    This is intentionally permissive; policies decide how to interpret fields.
    """
    version: Optional[str] = Field(default=None)
    task: Optional[str] = Field(default=None)

    # Example structure: metrics.<metric_name>.{min,max}
    metrics: Dict[str, ExtractMetricThreshold] = Field(default_factory=dict)

    # Optional knobs used by certain policies (e.g., minimum dataset size, etc.)
    params: Dict[str, Any] = Field(default_factory=dict)