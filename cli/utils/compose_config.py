# cli/utils/compose_config.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml  # requires pyyaml


def _as_env_lines(d: Mapping[str, str]) -> list[str]:
    lines: list[str] = []
    for k in sorted(d.keys()):
        v = d[k]
        if v is None:
            v = ""
        lines.append(f"{k}={v}")
    return lines


def _coerce_to_str_map(raw: Mapping[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in raw.items():
        key = str(k)
        if v is None:
            out[key] = ""
        elif isinstance(v, bool):
            out[key] = "1" if v else "0"
        else:
            out[key] = str(v)
    return out


def _normalize_profiles(profile: str | Sequence[str]) -> list[str]:
    if isinstance(profile, str):
        s = profile.strip()
        if not s:
            return []
        # allow "docker+jobs" or "docker,jobs"
        parts = [p.strip() for p in s.replace("+", ",").split(",")]
        return [p for p in parts if p]
    return [str(p).strip() for p in profile if str(p).strip()]


def render_compose_env_file(
    *,
    config_yaml_path: Path,
    profile: str | Sequence[str],
    out_env_path: Path,
    profiles_key: str = "profiles",
    extra_env: Mapping[str, str] | None = None,
) -> None:
    """
    Render a docker compose --env-file from YAML internal defaults.

    Supports merging multiple profiles. Merge order:
      profiles[0] -> profiles[1] -> ... -> profiles[n]
    with "last write wins".
    """
    if not config_yaml_path.exists():
        raise RuntimeError(f"Config YAML not found: {config_yaml_path}")

    data = yaml.safe_load(config_yaml_path.read_text()) or {}
    profiles_obj = (data or {}).get(profiles_key, {}) or {}
    if not isinstance(profiles_obj, dict):
        raise RuntimeError(f"'{profiles_key}' must be a mapping in {config_yaml_path}")

    wanted = _normalize_profiles(profile)
    if not wanted:
        raise RuntimeError("At least one defaults profile must be provided")

    available = sorted(map(str, profiles_obj.keys()))
    missing = [p for p in wanted if p not in profiles_obj]
    if missing:
        raise RuntimeError(
            f"Unknown profile(s) {missing} in {config_yaml_path} under '{profiles_key}'. "
            f"Available: {', '.join(available)}"
        )

    merged: dict[str, str] = {}
    for p in wanted:
        raw_env = profiles_obj.get(p) or {}
        if not isinstance(raw_env, dict):
            raise RuntimeError(f"Profile '{p}' must map to a dict of env vars")
        merged.update(_coerce_to_str_map(raw_env))

    if extra_env:
        for k, v in extra_env.items():
            merged[str(k)] = "" if v is None else str(v)

    out_env_path.parent.mkdir(parents=True, exist_ok=True)
    out_env_path.write_text("\n".join(_as_env_lines(merged)) + "\n")