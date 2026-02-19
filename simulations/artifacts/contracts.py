# simulations/artifacts/contracts.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union, cast

from llm_contracts.runtime.generate_slo import (
    RUNTIME_GENERATE_SLO_SCHEMA,
    GenerateSLOSnapshot,
    parse_generate_slo,
    write_generate_slo,
)
from llm_contracts.runtime.policy_decision import (
    POLICY_DECISION_SCHEMA_V2,
    PolicyDecisionSnapshot,
    parse_policy_decision,
    write_policy_decision,
)
from llm_contracts.schema import validate_internal

Pathish = Union[str, Path]


@dataclass(frozen=True)
class VerifyResult:
    ok: bool
    kind: str
    source_path: str
    error: Optional[str] = None
    snapshot: Optional[Any] = None


def validate_payload(schema_filename: str, payload: Dict[str, Any]) -> None:
    """
    Schema-only validation using the internal schema registry.
    Raises on failure.
    """
    validate_internal(schema_filename, payload)


def write_policy(path: Pathish, payload: Dict[str, Any]) -> Path:
    """
    Validate + atomic write of policy_decision_v2.
    """
    # write_policy_decision validates schema_version and schema itself
    return write_policy_decision(path, payload)


def write_slo(path: Pathish, payload: Dict[str, Any]) -> Path:
    """
    Validate + atomic write of runtime_generate_slo_v1.
    """
    # write_generate_slo validates schema itself
    return write_generate_slo(path, payload)


def verify_policy_payload(payload: Dict[str, Any], *, source_path: str = "<memory>") -> VerifyResult:
    """
    Parse-valid verification: schema + supported version + fail-closed semantics.
    """
    try:
        validate_payload(POLICY_DECISION_SCHEMA_V2, payload)
        snap = parse_policy_decision(payload, source_path=source_path)
        return VerifyResult(ok=True, kind="policy_decision_v2", source_path=source_path, snapshot=snap)
    except Exception as e:
        return VerifyResult(ok=False, kind="policy_decision_v2", source_path=source_path, error=f"{type(e).__name__}: {e}")


def verify_slo_payload(payload: Dict[str, Any], *, source_path: str = "<memory>") -> VerifyResult:
    """
    Parse-valid verification: schema + supported version + field extraction semantics.
    """
    try:
        validate_payload(RUNTIME_GENERATE_SLO_SCHEMA, payload)
        snap = parse_generate_slo(payload, source_path=source_path)
        return VerifyResult(ok=True, kind="runtime_generate_slo_v1", source_path=source_path, snapshot=snap)
    except Exception as e:
        return VerifyResult(ok=False, kind="runtime_generate_slo_v1", source_path=source_path, error=f"{type(e).__name__}: {e}")