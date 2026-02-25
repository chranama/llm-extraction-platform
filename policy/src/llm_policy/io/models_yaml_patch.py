# policy/src/llm_policy/io/models_yaml_patch.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import os

try:
    import yaml  # pyyaml
except Exception:  # pragma: no cover
    yaml = None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


_ALLOWED_STATUSES = {"unknown", "allowed", "blocked"}


@dataclass(frozen=True)
class PatchChange:
    scope: str  # "base" or f"profile:{name}" or "root" (legacy)
    model_id: str
    before: Optional[Any]
    after: Any


@dataclass(frozen=True)
class PatchResult:
    ok: bool
    changed: bool
    message: str
    path: str
    model_id: str
    enable: bool
    profile: Optional[str] = None
    changes: Tuple[PatchChange, ...] = ()


def _ensure_dict(x: Any) -> Dict[str, Any]:
    return x if isinstance(x, dict) else {}


def _ensure_list(x: Any) -> list[Any]:
    return x if isinstance(x, list) else []


def _get_model_id(entry: Any) -> Optional[str]:
    if not isinstance(entry, dict):
        return None
    for k in ("id", "model_id", "name"):
        v = entry.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _find_or_create_model_entry(models_list: list[Any], model_id: str) -> Dict[str, Any]:
    for it in models_list:
        if isinstance(it, dict) and _get_model_id(it) == model_id:
            return it
    new_entry: Dict[str, Any] = {"id": model_id}
    models_list.append(new_entry)
    return new_entry


# ---------------------------------------------------------------------
# Capabilities patching (minimal clamp-only)
# ---------------------------------------------------------------------


def _get_extract_capability_raw(entry: Dict[str, Any]) -> Any:
    caps = entry.get("capabilities")
    if not isinstance(caps, dict):
        return None
    return caps.get("extract")


def _set_extract_enabled(entry: Dict[str, Any], *, enable: bool) -> Any:
    """
    Clamp only. Keep it minimal:
      - bool is accepted by llm_config
      - dict w/ enabled is accepted by llm_config
    We default to bool for readability.
    """
    before = _get_extract_capability_raw(entry)

    caps = entry.get("capabilities")
    if not isinstance(caps, dict):
        caps = {}
        entry["capabilities"] = caps

    # If it was already an object with enabled, preserve object shape but only set enabled.
    if isinstance(before, dict):
        obj = dict(before)
        obj["enabled"] = bool(enable)
        # Do NOT keep/introduce assessed/provenance keys here (Option A puts those in assessment.*).
        for k in (
            "assessed",
            "assessed_at",
            "assessed_at_utc",
            "assessed_by",
            "assessed_pipeline",
            "eval_run_dir",
            "thresholds_profile",
            "deployment_key",
            "deployment",
            "status",
            "reason",
            "details",
        ):
            obj.pop(k, None)
        caps["extract"] = obj
    else:
        caps["extract"] = bool(enable)

    return before


# ---------------------------------------------------------------------
# Assessment patching (Option A source-of-truth)
# ---------------------------------------------------------------------


def _get_assessment_raw(entry: Dict[str, Any]) -> Any:
    v = entry.get("assessment")
    return v if isinstance(v, dict) else None


def _ensure_assessment(entry: Dict[str, Any]) -> Dict[str, Any]:
    blk = entry.get("assessment")
    if not isinstance(blk, dict):
        blk = {}
        entry["assessment"] = blk
    return blk


def _normalize_status_for_enable(enable: bool) -> str:
    # For onboarding, "enable_extract" is the gating decision. Map deterministically:
    return "allowed" if enable else "blocked"


def _set_assessment_block(
    entry: Dict[str, Any],
    *,
    enable: bool,
    assessed: bool,
    assessed_at_utc: str,
    assessed_by: Optional[str],
    assessed_pipeline: Optional[str],
    eval_run_dir: Optional[str],
    thresholds_profile: Optional[str],
    deployment: Optional[Dict[str, Any]],
    deployment_key: Optional[str],
    # Optional knobs:
    status: Optional[str] = None,
    reason: Optional[str] = None,
) -> Any:
    """
    Option A: write assessed/provenance into assessment.*.

    Minimal required fields for the demo:
      - assessment.assessed (bool)
      - assessment.status ("unknown"|"allowed"|"blocked")
      - assessment.assessed_at_utc (str)
      - assessment.deployment_key (str) [strongly recommended for determinism]
      - assessment.eval_run_dir (str) [recommended for traceability]

    We keep extra provenance keys too.
    """
    before = _get_assessment_raw(entry)
    blk = _ensure_assessment(entry)

    blk["assessed"] = bool(assessed)

    st = (status or "").strip().lower()
    if not st:
        st = _normalize_status_for_enable(bool(enable)) if assessed else "unknown"
    if st not in _ALLOWED_STATUSES:
        st = "unknown"
    blk["status"] = st

    blk["assessed_at_utc"] = str(assessed_at_utc)

    if isinstance(assessed_by, str) and assessed_by.strip():
        blk["assessed_by"] = assessed_by.strip()
    if isinstance(assessed_pipeline, str) and assessed_pipeline.strip():
        blk["assessed_pipeline"] = assessed_pipeline.strip()
    if isinstance(eval_run_dir, str) and eval_run_dir.strip():
        blk["eval_run_dir"] = eval_run_dir.strip()
    if isinstance(thresholds_profile, str) and thresholds_profile.strip():
        blk["thresholds_profile"] = thresholds_profile.strip()

    if isinstance(deployment_key, str) and deployment_key.strip():
        blk["deployment_key"] = deployment_key.strip()
    if isinstance(deployment, dict) and deployment:
        blk["deployment"] = dict(deployment)

    if isinstance(reason, str) and reason.strip():
        blk["reason"] = reason.strip()

    return before


def _snapshot_for_change_detection(entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deterministic snapshot for change detection.
    Treat these as material:
      - extract enabled clamp
      - assessment assessed/status
      - assessment deployment_key
    """
    cap = _get_extract_capability_raw(entry)
    if isinstance(cap, dict):
        cap_enabled = cap.get("enabled") if isinstance(cap.get("enabled"), bool) else None
    elif isinstance(cap, bool):
        cap_enabled = cap
    else:
        cap_enabled = None

    a = _get_assessment_raw(entry) or {}
    assessed_v = a.get("assessed") if isinstance(a.get("assessed"), bool) else None
    status_v = a.get("status") if isinstance(a.get("status"), str) else None
    dk_v = a.get("deployment_key") if isinstance(a.get("deployment_key"), str) else None

    return {
        "extract_enabled": cap_enabled,
        "assessed": assessed_v,
        "status": status_v,
        "deployment_key": dk_v,
    }


def _patch_models_block(
    block: Any,
    *,
    model_id: str,
    enable: bool,
    scope: str,
    assessed: bool,
    assessed_by: Optional[str],
    assessed_pipeline: Optional[str],
    eval_run_dir: Optional[str],
    thresholds_profile: Optional[str],
    deployment: Optional[Dict[str, Any]],
    deployment_key: Optional[str],
) -> list[PatchChange]:
    changes: list[PatchChange] = []
    if not isinstance(block, dict):
        return changes

    models_any = block.get("models")
    if not isinstance(models_any, list):
        models_any = []
        block["models"] = models_any

    entry = _find_or_create_model_entry(models_any, model_id)
    before_snap = _snapshot_for_change_detection(entry)

    assessed_at = _utc_now_iso()

    # 1) clamp
    _set_extract_enabled(entry, enable=enable)

    # 2) assessed/provenance (Option A)
    _set_assessment_block(
        entry,
        enable=enable,
        assessed=bool(assessed),
        assessed_at_utc=assessed_at,
        assessed_by=assessed_by,
        assessed_pipeline=assessed_pipeline,
        eval_run_dir=eval_run_dir,
        thresholds_profile=thresholds_profile,
        deployment=deployment,
        deployment_key=deployment_key,
        status=None,  # derived deterministically
        reason=None,
    )

    after_snap = _snapshot_for_change_detection(entry)

    # Material change?
    changed = before_snap != after_snap
    if changed:
        changes.append(PatchChange(scope=scope, model_id=model_id, before=before_snap, after=after_snap))

    return changes


def _select_profile_name_for_patch(doc: Dict[str, Any]) -> str:
    requested = (os.environ.get("MODELS_PROFILE") or "").strip()
    if not requested:
        requested = (os.environ.get("APP_PROFILE") or "").strip()
    if not requested:
        requested = "host-transformers"

    profiles = doc.get("profiles")
    if not isinstance(profiles, dict):
        return requested

    if requested in profiles:
        return requested
    if "host-transformers" in profiles:
        return "host-transformers"
    for k in profiles.keys():
        if isinstance(k, str) and k.strip():
            return k
    return requested


def patch_models_yaml_extract_capability(
    path: Path,
    model_id: str,
    enable: bool,
    *,
    profile: Optional[str] = None,
    deployment: Optional[Dict[str, Any]] = None,
    deployment_key: Optional[str] = None,
    assessed: bool = True,
    assessed_by: Optional[str] = None,
    assessed_pipeline: Optional[str] = None,
    eval_run_dir: Optional[str] = None,
    thresholds_profile: Optional[str] = None,
) -> PatchResult:
    """
    Patch config/models.yaml.

    Profiled models.yaml:
      - If profile provided: patch ONLY profiles.<profile>.models[]
      - Else: patch ONLY the selected profile (env-derived)

    Legacy models.yaml:
      - patch top-level models[]

    NOTE:
      - We still do NOT patch base.models by default (matches your design).
        To include base: MODELS_PATCH_INCLUDE_BASE=1
    """
    p = Path(path).expanduser()

    if yaml is None:
        return PatchResult(
            ok=False,
            changed=False,
            message="PyYAML is not available (import yaml failed). Add pyyaml to dependencies.",
            path=str(p),
            model_id=model_id,
            enable=bool(enable),
            profile=profile,
            changes=(),
        )

    if not p.exists():
        return PatchResult(
            ok=False,
            changed=False,
            message=f"models.yaml not found at {p}",
            path=str(p),
            model_id=model_id,
            enable=bool(enable),
            profile=profile,
            changes=(),
        )

    raw_text = p.read_text(encoding="utf-8")
    try:
        doc = yaml.safe_load(raw_text) or {}
    except Exception as e:
        return PatchResult(
            ok=False,
            changed=False,
            message=f"Failed to parse YAML: {type(e).__name__}: {e}",
            path=str(p),
            model_id=model_id,
            enable=bool(enable),
            profile=profile,
            changes=(),
        )

    if not isinstance(doc, dict):
        doc = {}

    changes: list[PatchChange] = []
    profiled = isinstance(doc.get("profiles"), dict) and isinstance(doc.get("base"), dict)

    if not profiled:
        changes.extend(
            _patch_models_block(
                doc,
                model_id=model_id,
                enable=enable,
                scope="root",
                assessed=bool(assessed),
                assessed_by=assessed_by,
                assessed_pipeline=assessed_pipeline,
                eval_run_dir=eval_run_dir,
                thresholds_profile=thresholds_profile or profile,
                deployment=deployment,
                deployment_key=deployment_key,
            )
        )
    else:
        profiles_any = doc.get("profiles")
        if not isinstance(profiles_any, dict):
            profiles_any = {}
            doc["profiles"] = profiles_any

        target_profile = (profile or "").strip() or _select_profile_name_for_patch(doc)

        prof_block = profiles_any.get(target_profile)
        if not isinstance(prof_block, dict):
            prof_block = {}
            profiles_any[target_profile] = prof_block

        changes.extend(
            _patch_models_block(
                prof_block,
                model_id=model_id,
                enable=enable,
                scope=f"profile:{target_profile}",
                assessed=bool(assessed),
                assessed_by=assessed_by,
                assessed_pipeline=assessed_pipeline,
                eval_run_dir=eval_run_dir,
                thresholds_profile=thresholds_profile or target_profile,
                deployment=deployment,
                deployment_key=deployment_key,
            )
        )

        if (os.environ.get("MODELS_PATCH_INCLUDE_BASE") or "").strip() == "1":
            base = doc.get("base")
            if not isinstance(base, dict):
                base = {}
                doc["base"] = base
            changes.extend(
                _patch_models_block(
                    base,
                    model_id=model_id,
                    enable=enable,
                    scope="base",
                    assessed=bool(assessed),
                    assessed_by=assessed_by,
                    assessed_pipeline=assessed_pipeline,
                    eval_run_dir=eval_run_dir,
                    thresholds_profile=thresholds_profile or target_profile,
                    deployment=deployment,
                    deployment_key=deployment_key,
                )
            )

        profile = target_profile

    changed = len(changes) > 0

    if changed:
        try:
            out = yaml.safe_dump(doc, sort_keys=False, allow_unicode=True)
            p.write_text(out, encoding="utf-8")
        except Exception as e:
            return PatchResult(
                ok=False,
                changed=False,
                message=f"Failed to write YAML: {type(e).__name__}: {e}",
                path=str(p),
                model_id=model_id,
                enable=bool(enable),
                profile=profile,
                changes=tuple(changes),
            )

    msg = "patched" if changed else "no changes (already in desired state)"
    return PatchResult(
        ok=True,
        changed=changed,
        message=msg,
        path=str(p),
        model_id=model_id,
        enable=bool(enable),
        profile=profile,
        changes=tuple(changes),
    )