# simulations/artifacts/policy_fixtures.py
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from simulations.artifacts.contracts import write_policy, verify_policy_payload


def _utc_now_iso() -> str:
    # Keep consistent with the rest of the repo: seconds precision, Z suffix is fine,
    # but policy_decision schema only requires date-time format.
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


# -----------------------------------------------------------------------------
# Demo A — generate_clamp_only (existing behavior, kept)
# -----------------------------------------------------------------------------


def build_generate_clamp_only_policy_payload(
    *,
    ok: bool,
    status: str,  # "allow" | "deny" | "unknown"
    generate_thresholds_profile: str,
    generate_max_new_tokens_cap: Optional[int],
    policy_name: str = "generate_slo_clamp",
    contract_errors: int = 0,
    model_id: Optional[str] = None,
    generated_at: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a schema-valid policy_decision_v2 payload for Demo A:
      pipeline = generate_clamp_only

    Contract rules enforced by schema for generate_clamp_only:
      - enable_extract must be null
      - thresholds_profile must be null
      - eval_run_dir must be null
      - generate_thresholds_profile is required
    """
    if status not in ("allow", "deny", "unknown"):
        raise ValueError("status must be one of: allow, deny, unknown")

    ts = generated_at or _utc_now_iso()

    payload: Dict[str, Any] = {
        "schema_version": "policy_decision_v2",
        "generated_at": ts,
        "policy": policy_name,
        "pipeline": "generate_clamp_only",
        "status": status,
        "ok": bool(ok),
        "enable_extract": None,  # required null for this pipeline
        "generate_max_new_tokens_cap": generate_max_new_tokens_cap,
        "contract_errors": int(contract_errors),
        "thresholds_profile": None,  # required null for this pipeline
        "thresholds_version": None,
        "generate_thresholds_profile": generate_thresholds_profile,
        "eval_run_dir": None,  # required null for this pipeline
        "eval_task": None,
        "eval_run_id": None,
        "model_id": model_id,
        "reasons": [],
        "warnings": [],
        "metrics": {},
        "contract_warnings": 0,
    }

    vr = verify_policy_payload(payload)
    if not vr.ok:
        raise ValueError(f"Policy payload failed verification: {vr.error}")

    return payload


def allow_no_clamp_policy(
    *,
    generate_thresholds_profile: str = "default",
    model_id: Optional[str] = None,
) -> Dict[str, Any]:
    return build_generate_clamp_only_policy_payload(
        ok=True,
        status="allow",
        generate_thresholds_profile=generate_thresholds_profile,
        generate_max_new_tokens_cap=None,
        model_id=model_id,
    )


def allow_with_clamp_policy(
    *,
    cap: int,
    generate_thresholds_profile: str = "default",
    model_id: Optional[str] = None,
) -> Dict[str, Any]:
    return build_generate_clamp_only_policy_payload(
        ok=True,
        status="allow",
        generate_thresholds_profile=generate_thresholds_profile,
        generate_max_new_tokens_cap=int(cap),
        model_id=model_id,
    )


def deny_policy(
    *,
    generate_thresholds_profile: str = "default",
    model_id: Optional[str] = None,
) -> Dict[str, Any]:
    return build_generate_clamp_only_policy_payload(
        ok=False,
        status="deny",
        generate_thresholds_profile=generate_thresholds_profile,
        generate_max_new_tokens_cap=None,
        model_id=model_id,
    )


def unknown_policy(
    *,
    generate_thresholds_profile: str = "default",
    model_id: Optional[str] = None,
) -> Dict[str, Any]:
    return build_generate_clamp_only_policy_payload(
        ok=False,
        status="unknown",
        generate_thresholds_profile=generate_thresholds_profile,
        generate_max_new_tokens_cap=None,
        model_id=model_id,
    )


# -----------------------------------------------------------------------------
# Demo B — extract_only (NEW)
# -----------------------------------------------------------------------------


def build_extract_only_policy_payload(
    *,
    ok: bool,
    status: str,  # "allow" | "deny" | "unknown"
    enable_extract: bool,
    thresholds_profile: str,
    eval_run_dir: str,
    policy_name: str = "extract_enablement",
    contract_errors: int = 0,
    thresholds_version: Optional[str] = None,
    eval_task: Optional[str] = None,
    eval_run_id: Optional[str] = None,
    model_id: Optional[str] = None,
    generated_at: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a schema-valid policy_decision_v2 payload for Demo B:
      pipeline = extract_only

    Contract rules enforced by schema for extract_only:
      - enable_extract is required (boolean)
      - thresholds_profile is required (string)
      - eval_run_dir is required (string)
      - generate_max_new_tokens_cap must be null
      - generate_thresholds_profile must be null
    """
    if status not in ("allow", "deny", "unknown"):
        raise ValueError("status must be one of: allow, deny, unknown")

    ts = generated_at or _utc_now_iso()

    payload: Dict[str, Any] = {
        "schema_version": "policy_decision_v2",
        "generated_at": ts,
        "policy": policy_name,
        "pipeline": "extract_only",
        "status": status,
        "ok": bool(ok),
        "enable_extract": bool(enable_extract),
        "generate_max_new_tokens_cap": None,  # required null for extract_only
        "contract_errors": int(contract_errors),
        "thresholds_profile": str(thresholds_profile),
        "thresholds_version": thresholds_version,
        "generate_thresholds_profile": None,  # required null for extract_only
        "eval_run_dir": str(eval_run_dir),
        "eval_task": eval_task,
        "eval_run_id": eval_run_id,
        "model_id": model_id,
        "reasons": [],
        "warnings": [],
        "metrics": {},
        "contract_warnings": 0,
    }

    # Compact (schema allows nulls, but we keep payload stable; do NOT drop keys)
    vr = verify_policy_payload(payload)
    if not vr.ok:
        raise ValueError(f"Policy payload failed verification: {vr.error}")

    return payload


def allow_extract_policy(
    *,
    thresholds_profile: str = "default",
    eval_run_dir: str,
    eval_task: Optional[str] = None,
    eval_run_id: Optional[str] = None,
    model_id: Optional[str] = None,
) -> Dict[str, Any]:
    return build_extract_only_policy_payload(
        ok=True,
        status="allow",
        enable_extract=True,
        thresholds_profile=thresholds_profile,
        eval_run_dir=eval_run_dir,
        eval_task=eval_task,
        eval_run_id=eval_run_id,
        model_id=model_id,
    )


def deny_extract_policy(
    *,
    thresholds_profile: str = "default",
    eval_run_dir: str,
    eval_task: Optional[str] = None,
    eval_run_id: Optional[str] = None,
    model_id: Optional[str] = None,
) -> Dict[str, Any]:
    # deny -> schema forces ok=false and enable_extract=false
    return build_extract_only_policy_payload(
        ok=False,
        status="deny",
        enable_extract=False,
        thresholds_profile=thresholds_profile,
        eval_run_dir=eval_run_dir,
        eval_task=eval_task,
        eval_run_id=eval_run_id,
        model_id=model_id,
    )


def unknown_extract_policy(
    *,
    thresholds_profile: str = "default",
    eval_run_dir: str,
    eval_task: Optional[str] = None,
    eval_run_id: Optional[str] = None,
    model_id: Optional[str] = None,
) -> Dict[str, Any]:
    # unknown -> schema forces ok=false and enable_extract=false
    return build_extract_only_policy_payload(
        ok=False,
        status="unknown",
        enable_extract=False,
        thresholds_profile=thresholds_profile,
        eval_run_dir=eval_run_dir,
        eval_task=eval_task,
        eval_run_id=eval_run_id,
        model_id=model_id,
    )


# -----------------------------------------------------------------------------
# Writers (shared)
# -----------------------------------------------------------------------------


def write_policy_latest(
    out_path: Path,
    payload: Dict[str, Any],
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return write_policy(out_path, payload)


def write_policy_latest_under_repo(
    repo_root: Path,
    payload: Dict[str, Any],
) -> Path:
    return write_policy_latest(repo_root / "policy_out" / "latest.json", payload)