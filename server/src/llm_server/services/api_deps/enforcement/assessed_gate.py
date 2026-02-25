# server/src/llm_server/services/api_deps/enforcement/assessed_gate.py
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Optional, Tuple

from fastapi import Request, status

from llm_server.core.errors import AppError
from llm_server.services.api_deps.core.models_config import models_config_from_request
from llm_server.services.api_deps.health.snapshots import deployment_metadata_snapshot

_ALLOWED_STATUSES = {"unknown", "allowed", "blocked"}


# ============================================================
# Deployment key (stable correlation id for eval/policy/runtime)
# ============================================================

def deployment_key_from_snapshot(dep: Dict[str, Any]) -> str:
    """
    Compute a stable deployment key from the deployment metadata snapshot.

    Never throws. Falls back to "unknown".
    """
    try:
        profiles = dep.get("profiles") if isinstance(dep, dict) else None
        platform_info = dep.get("platform") if isinstance(dep, dict) else None
        accelerators = dep.get("accelerators") if isinstance(dep, dict) else None
        container = dep.get("container") if isinstance(dep, dict) else None

        payload = {
            "profiles": profiles if isinstance(profiles, dict) else {},
            "container": bool(container),
            "platform": {
                "system": (platform_info or {}).get("system") if isinstance(platform_info, dict) else None,
                "machine": (platform_info or {}).get("machine") if isinstance(platform_info, dict) else None,
            },
            "accelerators": {
                "torch_present": bool((accelerators or {}).get("torch_present")) if isinstance(accelerators, dict) else False,
                "cuda_available": bool((accelerators or {}).get("cuda_available")) if isinstance(accelerators, dict) else False,
                "cuda_device_count": int((accelerators or {}).get("cuda_device_count") or 0) if isinstance(accelerators, dict) else 0,
                "mps_available": bool((accelerators or {}).get("mps_available")) if isinstance(accelerators, dict) else False,
            },
        }

        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    except Exception:
        return "unknown"


def deployment_key_from_request(request: Request) -> str:
    """
    Best-effort: never throws.
    """
    try:
        dep = deployment_metadata_snapshot(request)
        if not isinstance(dep, dict):
            return "unknown"
        k = dep.get("deployment_key")
        if isinstance(k, str) and k.strip():
            return k.strip()
        return deployment_key_from_snapshot(dep)
    except Exception:
        return "unknown"


# ============================================================
# models.yaml -> assessed snapshot
# ============================================================

def _get_model_spec(request: Request, model_id: str) -> Any | None:
    try:
        cfg = models_config_from_request(request)
        for sp in getattr(cfg, "models", []) or []:
            if getattr(sp, "id", None) == model_id:
                return sp
    except Exception:
        return None
    return None


def _cap_assessed_from_capability_raw(v: Any) -> Tuple[Optional[bool], Optional[str], Dict[str, Any]]:
    """
    Parse assessed semantics from a single capability raw value.

    Supported (best-effort):
      - bool: no assessed metadata
      - dict: may include
          assessed: bool
          status: "unknown"|"allowed"|"blocked"
          reason: str
          assessed_at_utc: str
          details: dict
    """
    if not isinstance(v, dict):
        return None, None, {}

    assessed = v.get("assessed")
    assessed_bool = assessed if isinstance(assessed, bool) else None

    st = v.get("status")
    st_norm = st.strip().lower() if isinstance(st, str) and st.strip() else None
    if st_norm is not None and st_norm not in _ALLOWED_STATUSES:
        st_norm = None

    extra: Dict[str, Any] = {}
    for k in ("reason", "assessed_at_utc", "assessed_at", "details"):
        if k in v:
            extra[k] = v.get(k)

    return assessed_bool, st_norm, extra


def _assessed_snapshot_from_models_yaml(*, request: Request, model_id: str, capability: str) -> Dict[str, Any]:
    """
    Source of truth: models.yaml (via parsed ModelsConfig on app.state / cached loader).

    Semantics:
      - required: defaults False unless explicitly asserted (model-level or cap-level)
      - status:
          * if required==False -> "allowed"
          * else if explicit status present -> use it
          * else if assessed==True -> "allowed"
          * else -> "unknown"
    """
    sp = _get_model_spec(request, model_id)
    deployment_key = getattr(sp, "deployment_key", None) if sp is not None else None

    # Defaults
    required = False
    assessed: Optional[bool] = None
    status_value: Optional[str] = None
    reason: Optional[str] = None
    assessed_at_utc: Optional[str] = None
    details: Dict[str, Any] = {}

    # Model-level assessment block
    assessment_blk = getattr(sp, "assessment", None) if sp is not None else None
    if isinstance(assessment_blk, dict):
        if isinstance(assessment_blk.get("require_assessed_gate"), bool):
            required = required or bool(assessment_blk["require_assessed_gate"])
        if isinstance(assessment_blk.get("required"), bool):
            required = required or bool(assessment_blk["required"])

        rf = assessment_blk.get("require_for")
        if isinstance(rf, (list, tuple, set)):
            rf_norm = {str(x).strip().lower() for x in rf if isinstance(x, str)}
            required = required or (capability.strip().lower() in rf_norm)

        if isinstance(assessment_blk.get("assessed"), bool):
            assessed = assessment_blk["assessed"]

        st = assessment_blk.get("status")
        if isinstance(st, str) and st.strip():
            st_norm = st.strip().lower()
            if st_norm in _ALLOWED_STATUSES:
                status_value = st_norm

        if isinstance(assessment_blk.get("reason"), str) and assessment_blk["reason"].strip():
            reason = assessment_blk["reason"].strip()
        if isinstance(assessment_blk.get("assessed_at_utc"), str) and assessment_blk["assessed_at_utc"].strip():
            assessed_at_utc = assessment_blk["assessed_at_utc"].strip()

        # best-effort extra details
        details.update({k: v for k, v in assessment_blk.items() if k not in ("require_assessed_gate", "required", "require_for")})

    # Capability-level raw metadata (preferred for per-capability assessed)
    cap_raw_map = getattr(sp, "capabilities_effective", None) if sp is not None else None
    if isinstance(cap_raw_map, dict):
        cap_raw_val = cap_raw_map.get(capability)
        cap_assessed, cap_status, cap_extra = _cap_assessed_from_capability_raw(cap_raw_val)

        if isinstance(cap_raw_val, dict):
            if isinstance(cap_raw_val.get("require_assessed_gate"), bool):
                required = required or bool(cap_raw_val["require_assessed_gate"])
            if isinstance(cap_raw_val.get("required"), bool):
                required = required or bool(cap_raw_val["required"])
            rf2 = cap_raw_val.get("require_for")
            if isinstance(rf2, (list, tuple, set)):
                rf2_norm = {str(x).strip().lower() for x in rf2 if isinstance(x, str)}
                required = required or (capability.strip().lower() in rf2_norm)

        if cap_assessed is not None:
            assessed = cap_assessed
        if cap_status is not None:
            status_value = cap_status

        if isinstance(cap_extra.get("reason"), str) and cap_extra["reason"].strip():
            reason = cap_extra["reason"].strip()

        for k in ("assessed_at_utc", "assessed_at"):
            v = cap_extra.get(k)
            if isinstance(v, str) and v.strip():
                assessed_at_utc = v.strip()
                break

        d = cap_extra.get("details")
        if isinstance(d, dict):
            details.update(d)
        else:
            if "details" in cap_extra:
                details["details"] = cap_extra.get("details")

    # Final status decision
    if not required:
        st_final = "allowed"
    else:
        if status_value in _ALLOWED_STATUSES:
            st_final = status_value
        elif assessed is True:
            st_final = "allowed"
        else:
            st_final = "unknown"

    return {
        "required": bool(required),
        "status": st_final,
        "selected_model_id": model_id,
        "selected_deployment_key": deployment_key if isinstance(deployment_key, str) and deployment_key.strip() else None,
        "assessed": assessed,
        "assessed_at_utc": assessed_at_utc,
        "reason": reason,
        "details": details,
        "source": "models.yaml",
    }


# ============================================================
# Enforcement
# ============================================================

def require_assessed_gate(
    *,
    request: Request,
    model_id: str,
    capability: str,
    selected_deployment_key: Optional[str] = None,
) -> None:
    """
    Enforce assessed-gate semantics sourced from models.yaml.

    Policy:
      - If required == False -> allow
      - If required == True:
          * status == "allowed" -> allow
          * status in ("unknown","blocked") -> reject 503
    """
    cap = (capability or "").strip().lower() or "extract"
    snap = _assessed_snapshot_from_models_yaml(request=request, model_id=model_id, capability=cap)
    required = bool(snap.get("required", False))

    if not required:
        return

    st = snap.get("status")
    st = st.strip().lower() if isinstance(st, str) else "unknown"
    if st == "allowed":
        return

    current_key = deployment_key_from_request(request)
    expected_key = (
        selected_deployment_key.strip()
        if isinstance(selected_deployment_key, str) and selected_deployment_key.strip()
        else (snap.get("selected_deployment_key") if isinstance(snap, dict) else None)
    )
    expected_key = expected_key.strip() if isinstance(expected_key, str) and expected_key.strip() else None

    reason = snap.get("reason")
    reason = reason.strip() if isinstance(reason, str) and reason.strip() else None

    raise AppError(
        code="assessed_gate_blocked",
        message="Model access blocked: assessed gate is not allowed.",
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        extra={
            "model_id": model_id,
            "capability": cap,
            "assessed_gate": snap,
            "deployment": {
                "current_deployment_key": current_key,
                "expected_deployment_key": expected_key,
                "deployment_key_matches": (expected_key == current_key) if expected_key else None,
            },
            "reason": reason,
        },
    )