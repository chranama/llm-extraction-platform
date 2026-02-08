# contracts/src/llm_contracts/runtime/policy_decision.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

from llm_contracts.schema import (
    atomic_write_json_internal,
    read_json_internal_versioned,
    validate_internal,
)

Pathish = Union[str, Path]

POLICY_DECISION_SCHEMA_V2 = "policy_decision_v2.schema.json"
SUPPORTED_POLICY_DECISION_VERSIONS = {"policy_decision_v2"}

_VERSION_TO_SCHEMA = {
    "policy_decision_v2": POLICY_DECISION_SCHEMA_V2,
}


@dataclass(frozen=True)
class PolicyDecisionSnapshot:
    """
    Parsed, validated, fail-closed view of a policy decision artifact (v2 only).
    """

    # Core validity
    ok: bool
    schema_version: str
    generated_at: str

    # Policy identity
    policy: str
    pipeline: str
    status: str  # allow | deny | unknown

    # Actions / shaping
    enable_extract: Optional[bool]
    generate_max_new_tokens_cap: Optional[int]

    # Contract health
    contract_errors: int

    # Provenance / traceability
    model_id: Optional[str]
    thresholds_profile: Optional[str]
    thresholds_version: Optional[str]
    generate_thresholds_profile: Optional[str]
    eval_run_dir: Optional[str]
    eval_task: Optional[str]
    eval_run_id: Optional[str]

    # Raw payload + IO context
    raw: Dict[str, Any]
    source_path: Optional[str] = None
    error: Optional[str] = None


def _opt_str(payload: Dict[str, Any], key: str) -> Optional[str]:
    v = payload.get(key)
    return v if isinstance(v, str) and v.strip() else None


def _opt_int(payload: Dict[str, Any], key: str) -> Optional[int]:
    v = payload.get(key)
    if v is None or isinstance(v, bool):
        return None
    return v if isinstance(v, int) else None


def parse_policy_decision(
    payload: Dict[str, Any],
    *,
    source_path: Optional[str] = None,
) -> PolicyDecisionSnapshot:
    """
    Parse and validate a policy decision artifact (v2 only).

    All fail-closed invariants are enforced here so downstream code
    can trust the snapshot.
    """
    schema_version = str(payload.get("schema_version", "")).strip()
    if schema_version not in SUPPORTED_POLICY_DECISION_VERSIONS:
        raise ValueError(f"Unsupported policy decision schema_version: {schema_version}")

    # Single source of truth: JSON Schema
    validate_internal(POLICY_DECISION_SCHEMA_V2, payload)

    generated_at = str(payload["generated_at"]).strip()
    policy = str(payload["policy"]).strip()
    pipeline = str(payload["pipeline"]).strip()
    status = str(payload["status"]).strip()

    ok = bool(payload["ok"])
    contract_errors = int(payload["contract_errors"])

    enable_extract = payload.get("enable_extract", None)
    if enable_extract is not None:
        enable_extract = bool(enable_extract)

    cap = _opt_int(payload, "generate_max_new_tokens_cap")
    if cap is not None and cap <= 0:
        cap = None

    # ----------------------------
    # Fail-closed invariants
    # ----------------------------
    if contract_errors > 0:
        ok = False
        enable_extract = False

    if status in ("deny", "unknown"):
        ok = False
        enable_extract = False

    if not ok:
        enable_extract = False

    return PolicyDecisionSnapshot(
        ok=ok,
        schema_version=schema_version,
        generated_at=generated_at,
        policy=policy,
        pipeline=pipeline,
        status=status,
        enable_extract=enable_extract,
        generate_max_new_tokens_cap=cap,
        contract_errors=contract_errors,
        model_id=_opt_str(payload, "model_id"),
        thresholds_profile=_opt_str(payload, "thresholds_profile"),
        thresholds_version=_opt_str(payload, "thresholds_version"),
        generate_thresholds_profile=_opt_str(payload, "generate_thresholds_profile"),
        eval_run_dir=_opt_str(payload, "eval_run_dir"),
        eval_task=_opt_str(payload, "eval_task"),
        eval_run_id=_opt_str(payload, "eval_run_id"),
        raw=dict(payload),
        source_path=source_path,
        error=None,
    )


def read_policy_decision(path: Pathish) -> PolicyDecisionSnapshot:
    """
    Read and parse a policy decision artifact from disk.

    Always returns a snapshot; failures become fail-closed snapshots.
    """
    p = Path(path).resolve()
    try:
        payload = read_json_internal_versioned(
            _VERSION_TO_SCHEMA,
            p,
            version_key="schema_version",
        )
        return parse_policy_decision(payload, source_path=str(p))
    except Exception as e:
        return _fail_closed_snapshot(p, e)


def _fail_closed_snapshot(p: Path, e: Exception) -> PolicyDecisionSnapshot:
    """
    Construct a synthetic fail-closed snapshot when parsing fails.
    """
    return PolicyDecisionSnapshot(
        ok=False,
        schema_version="policy_decision_v2",
        generated_at="",
        policy="",
        pipeline="unknown",
        status="unknown",
        enable_extract=False,
        generate_max_new_tokens_cap=None,
        contract_errors=1,
        model_id=None,
        thresholds_profile=None,
        thresholds_version=None,
        generate_thresholds_profile=None,
        eval_run_dir=None,
        eval_task=None,
        eval_run_id=None,
        raw={},
        source_path=str(p),
        error=f"policy_decision_parse_error: {type(e).__name__}: {e}",
    )


def write_policy_decision(path: Pathish, payload: Dict[str, Any]) -> Path:
    """
    Validate and atomically write a policy decision artifact (v2 only).
    """
    schema_version = str(payload.get("schema_version", "")).strip()
    if schema_version != "policy_decision_v2":
        raise ValueError(
            f"Unsupported policy decision schema_version: {schema_version} "
            "(only policy_decision_v2 is supported)"
        )

    return atomic_write_json_internal(POLICY_DECISION_SCHEMA_V2, path, payload)


def default_policy_out_path() -> Path:
    return (Path("policy_out") / "latest.json").resolve()