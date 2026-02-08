# policy/src/llm_policy/config.py
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

from llm_policy.types.extract_thresholds import ExtractThresholds
from llm_policy.types.generate_thresholds import GenerateThresholds
from llm_policy.utils.fs import read_yaml


@dataclass(frozen=True)
class PolicyConfig:
    """
    Policy runtime config.

    - thresholds_root: directory containing threshold profiles (e.g. thresholds/extract/*.yaml)
    """
    thresholds_root: str

    @staticmethod
    def default() -> "PolicyConfig":
        # package-relative: src/llm_policy/thresholds
        here = Path(__file__).resolve().parent
        root = str(here / "thresholds")
        env = os.getenv("LLM_POLICY_THRESHOLDS_ROOT")
        if env and env.strip():
            root = env.strip()
        return PolicyConfig(thresholds_root=root)


def _normalize_extract_profile(profile: Optional[str]) -> str:
    """
    Accept:
      - None -> "extract/default"
      - "extract/sroie"
      - "sroie" (shorthand -> "extract/sroie")
    """
    if not profile or not str(profile).strip():
        return "extract/default"

    p = str(profile).strip().replace("\\", "/").strip("/")
    if "/" not in p:
        p = f"extract/{p}"
    return p


def _normalize_generate_profile(profile: Optional[str]) -> str:
    """
    Accept:
      - None -> "generate/portable"
      - "generate/portable"
      - "portable" (shorthand -> "generate/portable")
    """
    if not profile or not str(profile).strip():
        return "generate/portable"

    p = str(profile).strip().replace("\\", "/").strip("/")
    if "/" not in p:
        p = f"generate/{p}"
    return p


def _load_thresholds_yaml(path: str) -> dict[str, Any]:
    obj = read_yaml(path)
    return obj if isinstance(obj, dict) else {}


def _safe_profile_path(*, root: Path, resolved: str) -> Path:
    """
    Build <root>/<resolved>.yaml and prevent traversal outside root.
    """
    root = root.resolve()
    yml_path = (root / f"{resolved}.yaml").resolve()

    if not str(yml_path).startswith(str(root)):
        raise ValueError("Invalid profile path (path traversal)")
    return yml_path


def load_extract_thresholds(
    *,
    cfg: Optional[PolicyConfig] = None,
    profile: Optional[str] = None,
) -> Tuple[str, ExtractThresholds]:
    """
    Load ExtractThresholds from:
      <thresholds_root>/<profile>.yaml

    Returns (resolved_profile, thresholds).
    """
    cfg = cfg or PolicyConfig.default()
    resolved = _normalize_extract_profile(profile)

    root = Path(cfg.thresholds_root)
    yml_path = _safe_profile_path(root=root, resolved=resolved)

    if not yml_path.exists():
        fallback = _safe_profile_path(root=root, resolved="extract/default")
        if not fallback.exists():
            raise FileNotFoundError(f"Thresholds file not found: {yml_path} (and no fallback {fallback})")
        obj = _load_thresholds_yaml(str(fallback))
        return "extract/default", ExtractThresholds.model_validate(obj)

    obj = _load_thresholds_yaml(str(yml_path))
    return resolved, ExtractThresholds.model_validate(obj)


def load_generate_thresholds(
    *,
    cfg: Optional[PolicyConfig] = None,
    profile: Optional[str] = None,
) -> Tuple[str, GenerateThresholds]:
    """
    Load GenerateThresholds from:
      <thresholds_root>/<profile>.yaml

    Returns (resolved_profile, thresholds).
    """
    cfg = cfg or PolicyConfig.default()
    resolved = _normalize_generate_profile(profile)

    root = Path(cfg.thresholds_root)
    yml_path = _safe_profile_path(root=root, resolved=resolved)

    if not yml_path.exists():
        fallback = _safe_profile_path(root=root, resolved="generate/portable")
        if not fallback.exists():
            raise FileNotFoundError(f"Generate thresholds file not found: {yml_path} (and no fallback {fallback})")
        obj = _load_thresholds_yaml(str(fallback))
        return "generate/portable", GenerateThresholds.model_validate(obj)

    obj = _load_thresholds_yaml(str(yml_path))
    return resolved, GenerateThresholds.model_validate(obj)