# policy/src/llm_policy/types/decision.py
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Literal

from pydantic import BaseModel, Field


# -------------------------
# Enums / literals
# -------------------------


class DecisionStatus(str, Enum):
    """
    Minimal tri-state:
      - allow: explicitly pass / enable
      - deny: explicitly fail / disable
      - unknown: policy could not determine (treat as deny for gating)
    """
    allow = "allow"
    deny = "deny"
    unknown = "unknown"


PipelineType = Literal[
    "extract_only",
    "generate_clamp_only",
    "extract_plus_generate_clamp",
]


# -------------------------
# Human-facing diagnostics
# -------------------------


class DecisionReason(BaseModel):
    code: str
    message: str
    context: Dict[str, Any] = Field(default_factory=dict)


class DecisionWarning(BaseModel):
    code: str
    message: str
    context: Dict[str, Any] = Field(default_factory=dict)


# -------------------------
# Canonical Decision object
# -------------------------


class Decision(BaseModel):
    """
    Canonical policy output object.

    This object is the *single in-memory representation* that flows through:
      - policy CLI
      - policy reports
      - runtime policy artifacts (policy_decision_v2)
      - backend ingestion

    Design rules:
      - Fail closed by default
      - Make pipeline selection explicit
      - Keep runtime shaping advisory (not gating)
    """

    # -------------------------
    # Identity
    # -------------------------

    policy: str = Field(
        ...,
        description="Stable policy identity for the emitted artifact (e.g. 'llm_policy')",
    )

    pipeline: PipelineType = Field(
        ...,
        description=(
            "Explicit policy pipeline selector. "
            "One of: extract_only | generate_clamp_only | extract_plus_generate_clamp"
        ),
    )

    status: DecisionStatus = Field(default=DecisionStatus.unknown)

    # -------------------------
    # Concrete actions
    # -------------------------

    # Gating action (used to patch models.yaml)
    enable_extract: bool = Field(default=False)

    # Runtime shaping (advisory; enforced by server)
    # None => no clamp
    generate_max_new_tokens_cap: Optional[int] = Field(
        default=None,
        description="Optional policy clamp for max_new_tokens in /v1/generate",
    )

    # -------------------------
    # Traceability / provenance
    # -------------------------

    thresholds_profile: Optional[str] = Field(default=None)
    thresholds_version: Optional[str] = Field(default=None)

    eval_run_dir: Optional[str] = Field(default=None)
    eval_task: Optional[str] = Field(default=None)
    eval_run_id: Optional[str] = Field(default=None)
    model_id: Optional[str] = Field(default=None)

    # -------------------------
    # Diagnostics
    # -------------------------

    reasons: List[DecisionReason] = Field(default_factory=list)
    warnings: List[DecisionWarning] = Field(default_factory=list)

    # Scalar metrics extracted from artifacts (JSON-serializable)
    metrics: Dict[str, Any] = Field(default_factory=dict)

    # Contract health (fail-closed)
    contract_errors: int = Field(default=0)
    contract_warnings: int = Field(default=0)

    # -------------------------
    # Semantics
    # -------------------------

    def ok(self) -> bool:
        """
        "ok" is used as:
          - CLI exit code
          - gating boolean
          - runtime signal

        Fail-closed:
          - contract errors => not ok
          - status != allow => not ok
        """
        if self.contract_errors > 0:
            return False
        return self.status == DecisionStatus.allow

    # -------------------------
    # Constructors (helpers)
    # -------------------------

    @classmethod
    def allow_extract(
        cls,
        *,
        policy: str,
        pipeline: PipelineType,
        thresholds_profile: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
        reasons: Optional[List[DecisionReason]] = None,
        warnings: Optional[List[DecisionWarning]] = None,
        generate_max_new_tokens_cap: Optional[int] = None,
        **kwargs: Any,
    ) -> "Decision":
        return cls(
            policy=policy,
            pipeline=pipeline,
            status=DecisionStatus.allow,
            enable_extract=True,
            generate_max_new_tokens_cap=generate_max_new_tokens_cap,
            thresholds_profile=thresholds_profile,
            metrics=metrics or {},
            reasons=reasons or [],
            warnings=warnings or [],
            **kwargs,
        )

    @classmethod
    def deny_extract(
        cls,
        *,
        policy: str,
        pipeline: PipelineType,
        thresholds_profile: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
        reasons: Optional[List[DecisionReason]] = None,
        warnings: Optional[List[DecisionWarning]] = None,
        generate_max_new_tokens_cap: Optional[int] = None,
        **kwargs: Any,
    ) -> "Decision":
        return cls(
            policy=policy,
            pipeline=pipeline,
            status=DecisionStatus.deny,
            enable_extract=False,
            generate_max_new_tokens_cap=generate_max_new_tokens_cap,
            thresholds_profile=thresholds_profile,
            metrics=metrics or {},
            reasons=reasons or [],
            warnings=warnings or [],
            **kwargs,
        )