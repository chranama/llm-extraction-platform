# server/src/llm_server/services/api_deps/health/snapshots.py
from __future__ import annotations

import os
import platform
import sys
import time
from typing import Any, Dict

from fastapi import Request

from llm_server.io.policy_decisions import get_policy_snapshot
from llm_server.services.api_deps.core.models_config import models_config_from_request
from llm_server.services.api_deps.core.policy_snapshot import policy_snapshot_summary
from llm_server.services.api_deps.routing.models import resolve_default_model_id_and_backend_obj
from llm_server.services.limits.generate_gating import get_generate_gate


# ============================================================
# Helpers
# ============================================================

def _utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _in_container_best_effort() -> bool:
    if os.getenv("IN_DOCKER", "").strip():
        return True
    try:
        p = "/proc/1/cgroup"
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                txt = f.read()
            return any(x in txt for x in ("docker", "kubepods", "containerd"))
    except Exception:
        pass
    return False


def _torch_accel_snapshot() -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "torch_present": False,
        "cuda_available": False,
        "cuda_device_count": 0,
        "mps_available": False,
    }

    try:
        import torch  # type: ignore

        out["torch_present"] = True

        try:
            out["cuda_available"] = bool(torch.cuda.is_available())
        except Exception:
            out["cuda_available"] = False

        try:
            out["cuda_device_count"] = int(torch.cuda.device_count()) if out["cuda_available"] else 0
        except Exception:
            out["cuda_device_count"] = 0

        try:
            mps = getattr(torch.backends, "mps", None)
            out["mps_available"] = bool(
                mps is not None
                and callable(getattr(mps, "is_available", None))
                and mps.is_available()
            )
        except Exception:
            out["mps_available"] = False

        return out

    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        return out


def _env_nonempty(name: str) -> str | None:
    v = os.getenv(name)
    if isinstance(v, str):
        s = v.strip()
        return s or None
    return None


def _backend_model_info_best_effort(backend_obj: Any) -> Dict[str, Any] | None:
    try:
        fn = getattr(backend_obj, "model_info", None)
        if callable(fn):
            info = fn()
            return info if isinstance(info, dict) else {"raw": info}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    return None


def _deployment_key_from_app_state(request: Request, *, model_id: str | None) -> str | None:
    env_override = _env_nonempty("DEPLOYMENT_KEY")
    if env_override:
        return env_override

    if not model_id:
        return None

    # Prefer registry meta (MultiModelManager)
    try:
        llm = getattr(request.app.state, "llm", None)
        meta = getattr(llm, "_meta", None)
        if isinstance(meta, dict):
            m = meta.get(model_id)
            if isinstance(m, dict):
                dk = m.get("deployment_key")
                if isinstance(dk, str) and dk.strip():
                    return dk.strip()
    except Exception:
        pass

    # Fallback: models_config
    try:
        cfg = getattr(request.app.state, "models_config", None)
        models = getattr(cfg, "models", None) if cfg is not None else None
        if isinstance(models, list):
            for sp in models:
                try:
                    mid = getattr(sp, "id", None)
                    if isinstance(mid, str) and mid == model_id:
                        dk = getattr(sp, "deployment_key", None)
                        if isinstance(dk, str) and dk.strip():
                            return dk.strip()
                        break
                except Exception:
                    continue
    except Exception:
        pass

    return None


# ============================================================
# Policy + Generate Gate
# ============================================================

def policy_summary(request: Request) -> Dict[str, Any]:
    snap = None
    try:
        snap = get_policy_snapshot(request)
    except Exception:
        snap = None
    return policy_snapshot_summary(snap)


def generate_gate_snapshot() -> Dict[str, Any]:
    try:
        gate = get_generate_gate()
        snap = gate.snapshot()
        return {
            "enabled": bool(snap.enabled),
            "max_concurrent": int(snap.max_concurrent),
            "max_queue": int(snap.max_queue),
            "timeout_seconds": float(snap.timeout_seconds),
            "in_flight_estimate": int(snap.in_flight_estimate),
            "queue_depth_estimate": int(snap.queue_depth_estimate),
        }
    except Exception:
        return {"error": "unavailable"}


# ============================================================
# Assessed Gate Snapshot (models.yaml sourced)
# ============================================================

_ALLOWED_STATUSES = {"unknown", "allowed", "blocked"}


def assessed_gate_snapshot(request: Request) -> Dict[str, Any]:
    """
    Snapshot of the assessed gate sourced from models.yaml (not in-memory).
    Best-effort and non-blocking.

    Shape:
      { ok, timestamp_utc, snapshot: {required,status,selected_model_id,...} }
    """
    try:
        default_model_id, _default_backend, _backend_obj = resolve_default_model_id_and_backend_obj(request)

        cfg = models_config_from_request(request)

        sp = None
        for m in getattr(cfg, "models", []) or []:
            if getattr(m, "id", None) == default_model_id:
                sp = m
                break

        if sp is None:
            snap = {
                "required": False,
                "status": "unknown",
                "selected_model_id": default_model_id,
                "selected_deployment_key": None,
                "source": "models.yaml",
                "error": "model_spec_not_found",
            }
            return {"ok": True, "timestamp_utc": _utc_now_iso(), "snapshot": snap}

        deployment_key = getattr(sp, "deployment_key", None)
        assessment_blk = getattr(sp, "assessment", None)

        required = False
        if isinstance(assessment_blk, dict):
            if isinstance(assessment_blk.get("require_assessed_gate"), bool):
                required = required or bool(assessment_blk["require_assessed_gate"])
            if isinstance(assessment_blk.get("required"), bool):
                required = required or bool(assessment_blk["required"])
            rf = assessment_blk.get("require_for")
            if isinstance(rf, (list, tuple, set)):
                rf_norm = {str(x).strip().lower() for x in rf if isinstance(x, str)}
                required = required or ("extract" in rf_norm)

        assessed = None
        status_value = None
        reason = None
        assessed_at_utc = None
        details: Dict[str, Any] = {}

        caps_eff = getattr(sp, "capabilities_effective", None)
        if isinstance(caps_eff, dict):
            v = caps_eff.get("extract")
            if isinstance(v, dict):
                if isinstance(v.get("assessed"), bool):
                    assessed = v["assessed"]

                st = v.get("status")
                if isinstance(st, str) and st.strip():
                    st_norm = st.strip().lower()
                    if st_norm in _ALLOWED_STATUSES:
                        status_value = st_norm

                if isinstance(v.get("reason"), str) and v["reason"].strip():
                    reason = v["reason"].strip()

                for k in ("assessed_at_utc", "assessed_at"):
                    vv = v.get(k)
                    if isinstance(vv, str) and vv.strip():
                        assessed_at_utc = vv.strip()
                        break

                d = v.get("details")
                if isinstance(d, dict):
                    details.update(d)

                # cap-level requirement flags (optional)
                if isinstance(v.get("require_assessed_gate"), bool):
                    required = required or bool(v["require_assessed_gate"])
                if isinstance(v.get("required"), bool):
                    required = required or bool(v["required"])

        if not required:
            st_final = "allowed"
        else:
            if status_value in _ALLOWED_STATUSES:
                st_final = status_value
            elif assessed is True:
                st_final = "allowed"
            else:
                st_final = "unknown"

        snap = {
            "required": bool(required),
            "status": st_final,
            "selected_model_id": default_model_id,
            "selected_deployment_key": deployment_key if isinstance(deployment_key, str) and deployment_key.strip() else None,
            "assessed": assessed,
            "assessed_at_utc": assessed_at_utc,
            "reason": reason,
            "details": details,
            "source": "models.yaml",
        }
        return {"ok": True, "timestamp_utc": _utc_now_iso(), "snapshot": snap}

    except Exception as e:
        return {"ok": False, "timestamp_utc": _utc_now_iso(), "error": f"{type(e).__name__}: {e}"}


# ============================================================
# Deployment Metadata Snapshot (with deployment_key)
# ============================================================

def deployment_metadata_snapshot(request: Request) -> Dict[str, Any]:
    try:
        app_profile = _env_nonempty("APP_PROFILE")
        models_profile_env = _env_nonempty("MODELS_PROFILE")

        models_profile_selected = None
        try:
            cfg = getattr(request.app.state, "models_config", None)
            defaults = getattr(cfg, "defaults", None) if cfg is not None else None
            if isinstance(defaults, dict):
                v = defaults.get("selected_profile")
                if isinstance(v, str) and v.strip():
                    models_profile_selected = v.strip()
        except Exception:
            models_profile_selected = None

        default_model_id, default_backend, backend_obj = resolve_default_model_id_and_backend_obj(request)

        backend_info = None
        if backend_obj is not None:
            backend_info = _backend_model_info_best_effort(backend_obj)

        deployment_key = _deployment_key_from_app_state(request, model_id=default_model_id)

        return {
            "ok": True,
            "timestamp_utc": _utc_now_iso(),
            "deployment_key": deployment_key,
            "identity": {
                "service_name": _env_nonempty("SERVICE_NAME"),
                "service_version": _env_nonempty("SERVICE_VERSION"),
                "git_sha": _env_nonempty("GIT_SHA"),
                "image_tag": _env_nonempty("IMAGE_TAG"),
                "build_time_utc": _env_nonempty("BUILD_TIME_UTC"),
            },
            "profiles": {
                "app_profile": app_profile,
                "models_profile_env": models_profile_env,
                "models_profile_selected": models_profile_selected,
            },
            "container": bool(_in_container_best_effort()),
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
                "python": sys.version.split()[0],
            },
            "accelerators": _torch_accel_snapshot(),
            "routing": {
                "default_model_id": default_model_id,
                "default_backend": default_backend,
                "backend_info": backend_info,
            },
        }

    except Exception as e:
        return {"ok": False, "timestamp_utc": _utc_now_iso(), "error": f"{type(e).__name__}: {e}"}