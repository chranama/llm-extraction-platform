# contracts/src/llm_contracts/config/models_config.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, cast

from pydantic import BaseModel, ConfigDict, Field


Backend = Literal["transformers", "llamacpp", "remote"]
LoadMode = Literal["eager", "lazy", "off"]
Capability = Literal["generate", "extract"]

_ALLOWED_BACKENDS = {"transformers", "llamacpp", "remote"}
_ALLOWED_LOAD_MODES = {"eager", "lazy", "off"}
_ALLOWED_CAP_KEYS = {"generate", "extract"}

_GENERIC_DEPLOYMENT_KEYS = {"default"}  # keep aligned with server enforcement


# ============================================================
# Validation result (lightweight, runtime-friendly)
# ============================================================

@dataclass(frozen=True)
class ValidationIssue:
    path: str
    message: str
    detail: Dict[str, Any]


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    issues: List[ValidationIssue]
    snapshot: Dict[str, Any]

    @property
    def error(self) -> str:
        if self.ok:
            return ""
        # compact one-liner summary
        head = self.issues[0] if self.issues else None
        if not head:
            return "validation failed"
        return f"{head.path}: {head.message}"


# ============================================================
# Snapshots (stable-ish structures for health/admin output)
# ============================================================

class ModelSpecSnapshot(BaseModel):
    """
    Snapshot of the normalized ModelSpec the server uses.

    Intentionally small and stable.
    """
    model_config = ConfigDict(extra="allow")

    id: str
    backend: Optional[str] = None
    load_mode: Optional[str] = None
    deployment_key: Optional[str] = None
    readiness_mode: Optional[str] = None

    # bool clamps after normalization (server uses this)
    capabilities: Optional[Dict[str, bool]] = None

    # Option A assessment block (pass-through)
    assessment: Optional[Dict[str, Any]] = None


class ModelsConfigSnapshot(BaseModel):
    model_config = ConfigDict(extra="allow")

    primary_id: str
    model_ids: List[str] = Field(default_factory=list)
    defaults: Dict[str, Any] = Field(default_factory=dict)
    models: List[ModelSpecSnapshot] = Field(default_factory=list)


# ============================================================
# Helpers: object/dict interoperability
# ============================================================

def _to_dict_best_effort(obj: Any) -> Dict[str, Any]:
    """
    Accept:
      - dict
      - pydantic model (model_dump)
      - dataclass-ish / attr object (getattr)
    """
    if isinstance(obj, dict):
        return obj

    # pydantic v2
    md = getattr(obj, "model_dump", None)
    if callable(md):
        try:
            d = md()
            return d if isinstance(d, dict) else {"raw": d}
        except Exception:
            pass

    # generic object with attrs
    out: Dict[str, Any] = {}
    for k in ("primary_id", "model_ids", "models", "defaults"):
        if hasattr(obj, k):
            try:
                out[k] = getattr(obj, k)
            except Exception:
                continue
    # include other attrs if someone passes a richer object
    return out


def _as_str(x: Any) -> Optional[str]:
    if isinstance(x, str):
        s = x.strip()
        return s or None
    return None


def _as_list_of_str(x: Any) -> List[str]:
    if not isinstance(x, list):
        return []
    out: List[str] = []
    for v in x:
        s = _as_str(v)
        if s:
            out.append(s)
    return out


def _models_list_best_effort(cfg_dict: Dict[str, Any]) -> List[Any]:
    m = cfg_dict.get("models")
    return m if isinstance(m, list) else []


def _snapshot(cfg: Any) -> Dict[str, Any]:
    d = _to_dict_best_effort(cfg)

    primary_id = _as_str(d.get("primary_id")) or ""
    model_ids = _as_list_of_str(d.get("model_ids"))

    defaults = d.get("defaults")
    defaults_dict = defaults if isinstance(defaults, dict) else {}

    models_snap: List[ModelSpecSnapshot] = []
    for item in _models_list_best_effort(d):
        if isinstance(item, dict):
            models_snap.append(ModelSpecSnapshot.model_validate(item))
        else:
            # attr object
            it: Dict[str, Any] = {}
            for k in ("id", "backend", "load_mode", "deployment_key", "readiness_mode", "capabilities", "assessment"):
                try:
                    if hasattr(item, k):
                        it[k] = getattr(item, k)
                except Exception:
                    continue
            models_snap.append(ModelSpecSnapshot.model_validate(it))

    snap = ModelsConfigSnapshot(
        primary_id=primary_id,
        model_ids=model_ids,
        defaults=defaults_dict,
        models=models_snap,
    )
    return snap.model_dump()


# ============================================================
# Core validators
# ============================================================

def validate_models_config(
    cfg: Any,
    *,
    allow_generic_deployment_key: bool = False,
) -> ValidationResult:
    """
    Validate the normalized ModelsConfig shape.

    This is meant to validate the *output* of server load_models_config(),
    not the raw YAML.

    Strict on:
      - ids
      - backend/load_mode enums
      - deployment_key presence (and non-generic unless allowed)
      - capabilities bool map keys
    """
    issues: List[ValidationIssue] = []
    snap = _snapshot(cfg)

    primary_id = _as_str(snap.get("primary_id"))
    if not primary_id:
        issues.append(
            ValidationIssue(
                path="primary_id",
                message="primary_id must be a non-empty string",
                detail={"value": snap.get("primary_id")},
            )
        )

    model_ids = snap.get("model_ids")
    if not isinstance(model_ids, list) or not model_ids:
        issues.append(
            ValidationIssue(
                path="model_ids",
                message="model_ids must be a non-empty list of strings",
                detail={"value": model_ids},
            )
        )
        model_ids_list: List[str] = []
    else:
        model_ids_list = [x for x in model_ids if isinstance(x, str) and x.strip()]
        if len(model_ids_list) != len(model_ids):
            issues.append(
                ValidationIssue(
                    path="model_ids",
                    message="model_ids contains empty/non-string entries",
                    detail={"value": model_ids},
                )
            )

    if primary_id and model_ids_list and primary_id not in model_ids_list:
        issues.append(
            ValidationIssue(
                path="model_ids",
                message="primary_id must be present in model_ids",
                detail={"primary_id": primary_id, "model_ids": model_ids_list},
            )
        )

    models = snap.get("models")
    if not isinstance(models, list) or not models:
        issues.append(
            ValidationIssue(
                path="models",
                message="models must be a non-empty list",
                detail={"value": models},
            )
        )
        models_list: List[Dict[str, Any]] = []
    else:
        models_list = [m for m in models if isinstance(m, dict)]

    # Validate each model spec
    seen: set[str] = set()
    for i, m in enumerate(models_list):
        mid = _as_str(m.get("id"))
        pfx = f"models[{i}]"

        if not mid:
            issues.append(ValidationIssue(path=f"{pfx}.id", message="id must be a non-empty string", detail={"value": m.get("id")}))
            continue

        if mid in seen:
            issues.append(ValidationIssue(path=f"{pfx}.id", message="duplicate model id", detail={"id": mid}))
        seen.add(mid)

        backend = _as_str(m.get("backend"))
        if backend and backend not in _ALLOWED_BACKENDS:
            issues.append(
                ValidationIssue(
                    path=f"{pfx}.backend",
                    message="backend must be one of: transformers|llamacpp|remote",
                    detail={"backend": backend, "allowed": sorted(_ALLOWED_BACKENDS)},
                )
            )

        load_mode = _as_str(m.get("load_mode"))
        if load_mode and load_mode not in _ALLOWED_LOAD_MODES:
            issues.append(
                ValidationIssue(
                    path=f"{pfx}.load_mode",
                    message="load_mode must be one of: eager|lazy|off",
                    detail={"load_mode": load_mode, "allowed": sorted(_ALLOWED_LOAD_MODES)},
                )
            )

        dk = _as_str(m.get("deployment_key"))
        if not dk:
            issues.append(
                ValidationIssue(
                    path=f"{pfx}.deployment_key",
                    message="deployment_key must be a non-empty string",
                    detail={"model_id": mid, "value": m.get("deployment_key")},
                )
            )
        else:
            if (not allow_generic_deployment_key) and (dk.strip().lower() in _GENERIC_DEPLOYMENT_KEYS):
                issues.append(
                    ValidationIssue(
                        path=f"{pfx}.deployment_key",
                        message="deployment_key must not be generic",
                        detail={"model_id": mid, "deployment_key": dk, "generic": sorted(_GENERIC_DEPLOYMENT_KEYS)},
                    )
                )

        caps = m.get("capabilities")
        if caps is not None:
            if not isinstance(caps, dict):
                issues.append(
                    ValidationIssue(
                        path=f"{pfx}.capabilities",
                        message="capabilities must be a mapping of {generate,extract} -> bool (if provided)",
                        detail={"model_id": mid, "value": caps},
                    )
                )
            else:
                for k, v in caps.items():
                    if not isinstance(k, str) or not k.strip():
                        issues.append(
                            ValidationIssue(
                                path=f"{pfx}.capabilities",
                                message="capabilities keys must be strings",
                                detail={"model_id": mid, "key": k},
                            )
                        )
                        continue
                    kk = k.strip()
                    if kk not in _ALLOWED_CAP_KEYS:
                        issues.append(
                            ValidationIssue(
                                path=f"{pfx}.capabilities.{kk}",
                                message="invalid capability key",
                                detail={"model_id": mid, "key": kk, "allowed": sorted(_ALLOWED_CAP_KEYS)},
                            )
                        )
                    if not isinstance(v, bool):
                        issues.append(
                            ValidationIssue(
                                path=f"{pfx}.capabilities.{kk}",
                                message="capability value must be boolean",
                                detail={"model_id": mid, "key": kk, "value": v},
                            )
                        )

    ok = len(issues) == 0
    return ValidationResult(ok=ok, issues=issues, snapshot=snap)


def validate_assessment_for_extract(
    cfg: Any,
    *,
    bypass_if_profile_test: bool = True,
) -> ValidationResult:
    """
    Option A enforcement contract:
      - if defaults.assessment.require_for_extract is true:
          every model must have assessment.assessed as boolean
          (typically false in demo/unassessed state)

    Notes:
      - Reads selected profile from cfg.defaults.selected_profile if present.
      - Tolerates missing/default blocks when require_for_extract is false.
    """
    issues: List[ValidationIssue] = []
    snap = _snapshot(cfg)

    defaults = snap.get("defaults") if isinstance(snap.get("defaults"), dict) else {}
    selected_profile = None
    if isinstance(defaults, dict):
        selected_profile = _as_str(defaults.get("selected_profile"))

    if bypass_if_profile_test and selected_profile and selected_profile.strip().lower() == "test":
        return ValidationResult(ok=True, issues=[], snapshot=snap)

    assessment_defaults = None
    if isinstance(defaults, dict):
        assessment_defaults = defaults.get("assessment")

    require_for_extract = False
    if isinstance(assessment_defaults, dict):
        require_for_extract = bool(assessment_defaults.get("require_for_extract", False))

    # If extract assessment isn’t required, we’re done.
    if not require_for_extract:
        return ValidationResult(ok=True, issues=[], snapshot=snap)

    models = snap.get("models")
    if not isinstance(models, list) or not models:
        issues.append(
            ValidationIssue(
                path="models",
                message="models must be a non-empty list when require_for_extract is true",
                detail={"value": models},
            )
        )
        return ValidationResult(ok=False, issues=issues, snapshot=snap)

    for i, m in enumerate(models):
        if not isinstance(m, dict):
            continue
        mid = _as_str(m.get("id")) or f"<unknown:{i}>"
        pfx = f"models[{i}]"

        assessment = m.get("assessment")
        if not isinstance(assessment, dict):
            issues.append(
                ValidationIssue(
                    path=f"{pfx}.assessment",
                    message="assessment block must be present (mapping) when require_for_extract is true",
                    detail={"model_id": mid, "value": assessment},
                )
            )
            continue

        assessed = assessment.get("assessed")
        if not isinstance(assessed, bool):
            issues.append(
                ValidationIssue(
                    path=f"{pfx}.assessment.assessed",
                    message="assessment.assessed must be a boolean when require_for_extract is true",
                    detail={"model_id": mid, "value": assessed, "assessment": assessment},
                )
            )

    ok = len(issues) == 0
    return ValidationResult(ok=ok, issues=issues, snapshot=snap)