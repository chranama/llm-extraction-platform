# simulations/artifacts/models_yaml_demo.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class DemoModelsYamlError(Exception):
    pass


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _ensure_dict(x: Any) -> Dict[str, Any]:
    return x if isinstance(x, dict) else {}


def _ensure_list(x: Any) -> list[Any]:
    return x if isinstance(x, list) else []


def _get_model_id(entry: Any) -> Optional[str]:
    if isinstance(entry, str):
        s = entry.strip()
        return s or None
    if not isinstance(entry, dict):
        return None
    v = entry.get("id")
    if isinstance(v, str) and v.strip():
        return v.strip()
    return None


def _find_model_entry(models_list: list[Any], model_id: str) -> Optional[Dict[str, Any]]:
    for it in models_list:
        if isinstance(it, dict) and _get_model_id(it) == model_id:
            return it
    return None


# ============================================================
# Compliance helpers (Option A)
# - assessed is read from models.yaml under `assessment.assessed` (NOT capabilities.*)
# ============================================================

def _ensure_assessment_block(entry: Dict[str, Any]) -> Dict[str, Any]:
    blk = entry.get("assessment")
    if not isinstance(blk, dict):
        blk = {}
        entry["assessment"] = blk
    return blk


def _normalize_assessment_assessed(entry: Dict[str, Any], *, assessed: bool = False) -> None:
    """
    Ensure `assessment.assessed` exists and is boolean.
    This is the compliance requirement for the assessed wiring.
    """
    assessment = _ensure_assessment_block(entry)
    assessment["assessed"] = bool(assessed)


def _normalize_extract_enabled(entry: Dict[str, Any], *, enabled: bool = False) -> None:
    """
    Set capabilities.extract to a boolean clamp.
    (We do NOT carry assessed metadata here anymore.)
    """
    caps = entry.get("capabilities")
    if not isinstance(caps, dict):
        caps = {}
        entry["capabilities"] = caps
    caps["extract"] = bool(enabled)


def _normalize_generate_default(entry: Dict[str, Any], *, enabled: bool = True) -> None:
    """
    Ensure generate is explicitly set (helps avoid surprises if defaults change).
    """
    caps = entry.get("capabilities")
    if not isinstance(caps, dict):
        caps = {}
        entry["capabilities"] = caps
    if "generate" not in caps:
        caps["generate"] = bool(enabled)


@dataclass(frozen=True)
class BuildResult:
    out_path: Path
    profile: str
    model_id: str
    source_path: Path


def build_demo_models_yaml(
    *,
    src_models_yaml: Path,
    out_models_yaml: Path,
    profile: str,
    model_id: str,
    assessed: bool = False,
    extract_enabled: bool = False,
    keep_base_defaults: bool = True,
    add_demo_metadata: bool = True,
) -> BuildResult:
    """
    Create a deterministic demo models.yaml (profiled shape):
      - ONLY the selected profile is included
      - ONLY the selected model entry exists in that profile
      - `assessment.assessed` is explicitly set to <assessed> (Option A compliance)
      - `capabilities.extract` is set to <extract_enabled> (bool clamp)
      - `capabilities.generate` is ensured present

    This is meant to be consumed by policy patching during a demo run.
    """
    src_models_yaml = Path(src_models_yaml).expanduser()
    out_models_yaml = Path(out_models_yaml).expanduser()
    profile = (profile or "").strip()
    model_id = (model_id or "").strip()

    if not profile:
        raise DemoModelsYamlError("profile is required")
    if not model_id:
        raise DemoModelsYamlError("model_id is required")
    if not src_models_yaml.exists():
        raise DemoModelsYamlError(f"source models.yaml not found: {src_models_yaml}")

    doc = yaml.safe_load(src_models_yaml.read_text(encoding="utf-8")) or {}
    if not isinstance(doc, dict):
        raise DemoModelsYamlError("source models.yaml must be a mapping")

    base = doc.get("base")
    profiles = doc.get("profiles")

    # Require profiled shape so we don't accidentally generate a legacy doc
    if not isinstance(base, dict) or not isinstance(profiles, dict):
        raise DemoModelsYamlError("source models.yaml is not in profiled shape (missing base/profiles dicts)")

    prof_block_any = profiles.get(profile)
    if not isinstance(prof_block_any, dict):
        available = sorted([k for k in profiles.keys() if isinstance(k, str)])
        raise DemoModelsYamlError(f"profile not found: {profile}. available={available}")

    prof_models = _ensure_list(prof_block_any.get("models"))
    entry = _find_model_entry(prof_models, model_id)
    if entry is None:
        available_ids = sorted([_get_model_id(x) for x in prof_models if _get_model_id(x)])
        raise DemoModelsYamlError(f"model_id not found in profile {profile}: {model_id}. available={available_ids}")

    # Build output
    out_doc: Dict[str, Any] = {}

    # base scaffold
    if keep_base_defaults:
        out_doc["base"] = dict(base)
    else:
        out_doc["base"] = {"defaults": _ensure_dict(base.get("defaults"))}

    # IMPORTANT: inventory is per-profile; keep base.models empty
    out_doc["base"]["models"] = []

    # profiles: only one
    out_profile_block = dict(prof_block_any)

    # isolate the one model entry
    new_entry = dict(entry)

    # enforce compliance + deterministic knobs
    _normalize_assessment_assessed(new_entry, assessed=assessed)
    _normalize_extract_enabled(new_entry, enabled=extract_enabled)
    _normalize_generate_default(new_entry, enabled=True)

    out_profile_block["models"] = [new_entry]
    out_doc["profiles"] = {profile: out_profile_block}

    if add_demo_metadata:
        out_doc["demo"] = {
            "kind": "demo_models_yaml",
            "generated_at": _utc_now_iso(),
            "source_models_yaml": str(src_models_yaml),
            "profile": profile,
            "model_id": model_id,
            "assessed": bool(assessed),
            "extract_enabled": bool(extract_enabled),
        }

    out_models_yaml.parent.mkdir(parents=True, exist_ok=True)
    out_models_yaml.write_text(
        yaml.safe_dump(out_doc, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    return BuildResult(out_path=out_models_yaml, profile=profile, model_id=model_id, source_path=src_models_yaml)