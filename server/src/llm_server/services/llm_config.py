# server/src/llm_server/services/llm_config.py
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import yaml

from llm_server.core.config import get_settings
from llm_server.core.errors import AppError

logger = logging.getLogger("llm_server.models")

# -----------------------------
# Types / allowed values
# -----------------------------

Backend = Literal["transformers", "llamacpp", "remote"]
LoadMode = Literal["eager", "lazy", "off"]

Capability = Literal["generate", "extract"]
CapabilitiesMap = Dict[Capability, bool]

_ALLOWED_BACKENDS = {"transformers", "llamacpp", "remote"}
_ALLOWED_LOAD_MODES = {"eager", "lazy", "off"}
_ALLOWED_CAP_KEYS: set[str] = {"generate", "extract"}


# -----------------------------
# Normalized config objects
# -----------------------------


@dataclass(frozen=True)
class ModelSpec:
    id: str
    backend: Backend = "transformers"
    load_mode: LoadMode = "lazy"
    capabilities: Optional[CapabilitiesMap] = None

    # backend config blocks (pass-through dicts)
    transformers: Optional[Dict[str, Any]] = None
    llamacpp: Optional[Dict[str, Any]] = None
    remote: Optional[Dict[str, Any]] = None

    notes: Optional[str] = None


@dataclass(frozen=True)
class ModelsConfig:
    primary_id: str
    model_ids: List[str]
    models: List[ModelSpec]
    defaults: Dict[str, Any]


# -----------------------------
# Helpers
# -----------------------------


def _app_root() -> Path:
    v = (os.environ.get("APP_ROOT") or "").strip()
    return Path(v).expanduser().resolve() if v else Path.cwd().resolve()


def _resolve_path_maybe_relative(path: str) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else (_app_root() / p).resolve()


def _resolve_models_yaml_path() -> str:
    env_path = (os.environ.get("MODELS_YAML") or "").strip()
    if env_path:
        return str(_resolve_path_maybe_relative(env_path))

    s = get_settings()
    raw = str(getattr(s, "models_config_path", None) or "config/models.yaml").strip()
    return str(_resolve_path_maybe_relative(raw or "config/models.yaml"))


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for x in items:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


# -----------------------------
# ${ENV_VAR} expansion
# -----------------------------

# supports:
#   ${VAR} -> env VAR or ""
#   ${VAR:-default} -> env VAR or "default"
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


def _expand_env_in_str(s: str) -> str:
    def repl(m: re.Match) -> str:
        key = m.group(1)
        default = m.group(2)
        val = os.environ.get(key)
        if val is None or not str(val).strip():
            return (default or "")
        return str(val)

    # expand repeatedly in case defaults contain more ${...}
    prev = None
    cur = s
    for _ in range(5):
        prev = cur
        cur = _ENV_PATTERN.sub(repl, cur)
        if cur == prev:
            break
    return cur


def _expand_env(obj: Any) -> Any:
    """
    Recursively expand ${VAR} in all strings within nested dict/list structures.
    """
    if isinstance(obj, str):
        return _expand_env_in_str(obj)
    if isinstance(obj, list):
        return [_expand_env(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(_expand_env(x) for x in obj)
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    return obj


# -----------------------------
# Merge logic (models-aware)
# -----------------------------


def _deep_merge_dicts(base: Any, overlay: Any) -> Any:
    if not isinstance(base, dict) or not isinstance(overlay, dict):
        return overlay
    out = dict(base)
    for k, v in overlay.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge_dicts(out[k], v)
        else:
            out[k] = v
    return out


def _model_id_of(item: Any) -> Optional[str]:
    if isinstance(item, str):
        s = item.strip()
        return s or None
    if isinstance(item, dict):
        mid = item.get("id")
        if isinstance(mid, str) and mid.strip():
            return mid.strip()
    return None


def _merge_models_list_by_id(base_list: Any, overlay_list: Any) -> Any:
    if overlay_list is None:
        return base_list
    if base_list is None:
        return overlay_list
    if not isinstance(base_list, list) or not isinstance(overlay_list, list):
        return overlay_list

    base_order: List[str] = []
    base_map: Dict[str, Any] = {}
    base_unknown: List[Any] = []

    for item in base_list:
        mid = _model_id_of(item)
        if not mid:
            base_unknown.append(item)
            continue
        if mid not in base_map:
            base_order.append(mid)
            base_map[mid] = item

    overlay_order_new: List[str] = []
    for item in overlay_list:
        mid = _model_id_of(item)
        if not mid:
            continue
        if mid in base_map:
            a = base_map[mid]
            b = item
            if isinstance(a, dict) and isinstance(b, dict):
                base_map[mid] = _deep_merge_dicts(a, b)
            else:
                base_map[mid] = b
        else:
            base_map[mid] = item
            overlay_order_new.append(mid)

    merged: List[Any] = []
    for mid in base_order:
        merged.append(base_map[mid])
    for mid in overlay_order_new:
        merged.append(base_map[mid])
    merged.extend(base_unknown)
    return merged


def _deep_merge_models_aware(base: Any, overlay: Any) -> Any:
    if not isinstance(base, dict) or not isinstance(overlay, dict):
        return overlay

    out = dict(base)
    for k, v in overlay.items():
        if k == "models":
            out[k] = _merge_models_list_by_id(out.get(k), v)
            continue
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge_models_aware(out[k], v)
        else:
            out[k] = v
    return out


def _select_profile(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Supports:
      1) legacy: {default_model, defaults, models}
      2) profiled: {base, profiles{...}}

    Selection:
      MODELS_PROFILE env var selects profile (preferred).
      Fallback to APP_PROFILE, then "host".
    """
    if not isinstance(raw, dict):
        return {}

    if "profiles" not in raw and "base" not in raw:
        return raw  # legacy

    base = raw.get("base") or {}
    profiles = raw.get("profiles") or {}
    if not isinstance(base, dict):
        base = {}
    if not isinstance(profiles, dict):
        profiles = {}

    requested = (os.environ.get("MODELS_PROFILE") or "").strip()
    source = "MODELS_PROFILE"
    if not requested:
        requested = (os.environ.get("APP_PROFILE") or "").strip()
        source = "APP_PROFILE"
    if not requested:
        requested = "host"
        source = "default"

    used = requested
    overlay = profiles.get(used)

    if overlay is None:
        fallback = "host" if "host" in profiles else (next(iter(profiles.keys()), "") or "")
        if fallback:
            logger.warning(
                "models: requested profile missing; falling back: requested=%r source=%s fallback=%r available=%s",
                requested,
                source,
                fallback,
                sorted(list(profiles.keys())),
            )
            used = fallback
            overlay = profiles.get(used) or {}
        else:
            logger.warning(
                "models: requested profile missing and no profiles available; using base only: requested=%r source=%s",
                requested,
                source,
            )
            used = requested
            overlay = {}

    if not isinstance(overlay, dict):
        overlay = {}

    merged = _deep_merge_models_aware(base, overlay)
    merged["_selected_profile_requested"] = requested
    merged["_selected_profile_source"] = source
    merged["_selected_profile_used"] = used
    return merged


# -----------------------------
# Validation helpers
# -----------------------------


def _as_str(x: Any, *, field: str, path: str) -> str:
    if not isinstance(x, str) or not x.strip():
        raise AppError(
            code="models_yaml_invalid",
            message=f"models.yaml {field} must be a non-empty string",
            status_code=500,
            extra={"path": path, "field": field, "value": x},
        )
    return x.strip()


def _as_opt_str(x: Any) -> Optional[str]:
    if x is None:
        return None
    if isinstance(x, str):
        s = x.strip()
        return s if s else None
    return None


def _validate_enum(
    value: Any,
    *,
    field: str,
    path: str,
    allowed: set[Any],
    coerce_lower: bool = True,
) -> Any:
    if value is None:
        return None
    if isinstance(value, str) and coerce_lower:
        value = value.strip().lower()
    if value not in allowed:
        raise AppError(
            code="models_yaml_invalid",
            message=f"models.yaml {field} has invalid value",
            status_code=500,
            extra={"path": path, "field": field, "value": value, "allowed": sorted(list(allowed))},
        )
    return value


def _load_yaml(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        raise AppError(
            code="models_yaml_missing",
            message="models.yaml not found",
            status_code=500,
            extra={"path": path},
        )
    except Exception as e:
        raise AppError(
            code="models_yaml_invalid",
            message="Failed to read models.yaml",
            status_code=500,
            extra={"path": path, "error": str(e)},
        ) from e

    if not data or not isinstance(data, dict):
        raise AppError(
            code="models_yaml_invalid",
            message="models.yaml must be a non-empty mapping (dict)",
            status_code=500,
            extra={"path": path},
        )
    return data


def _normalize_capabilities(raw_caps: Any, *, path: str, field: str) -> Optional[CapabilitiesMap]:
    if raw_caps is None:
        return None
    if not isinstance(raw_caps, dict):
        raise AppError(
            code="models_yaml_invalid",
            message=f"models.yaml {field} must be a mapping (dict)",
            status_code=500,
            extra={"path": path, "field": field, "value": raw_caps},
        )

    out: Dict[str, bool] = {}
    for k, v in raw_caps.items():
        if not isinstance(k, str) or not k.strip():
            raise AppError(
                code="models_yaml_invalid",
                message=f"models.yaml {field} keys must be strings",
                status_code=500,
                extra={"path": path, "field": field, "key": k},
            )
        kk = k.strip()
        if kk not in _ALLOWED_CAP_KEYS:
            raise AppError(
                code="models_yaml_invalid",
                message=f"models.yaml {field} has invalid capability key",
                status_code=500,
                extra={"path": path, "field": field, "key": kk, "allowed": sorted(list(_ALLOWED_CAP_KEYS))},
            )
        if not isinstance(v, bool):
            raise AppError(
                code="models_yaml_invalid",
                message=f"models.yaml {field}.{kk} must be a boolean",
                status_code=500,
                extra={"path": path, "field": f"{field}.{kk}", "value": v},
            )
        out[kk] = v

    return out  # type: ignore[return-value]


def _normalize_backend_block(raw_block: Any, *, path: str, field: str) -> Optional[Dict[str, Any]]:
    if raw_block is None:
        return None
    if not isinstance(raw_block, dict):
        raise AppError(
            code="models_yaml_invalid",
            message=f"models.yaml {field} must be a mapping (dict)",
            status_code=500,
            extra={"path": path, "field": field, "value": raw_block},
        )
    return dict(raw_block)


def _normalize_model_entry(raw: Any, *, path: str, defaults: Dict[str, Any]) -> ModelSpec:
    if isinstance(raw, str):
        mid = _as_str(raw, field="models[]", path=path)
        return ModelSpec(
            id=mid,
            backend=defaults["backend"],
            load_mode=defaults["load_mode"],
            capabilities=defaults.get("capabilities"),
            transformers=dict(defaults.get("transformers") or {}) or None,
            llamacpp=dict(defaults.get("llamacpp") or {}) or None,
            remote=dict(defaults.get("remote") or {}) or None,
            notes=None,
        )

    if not isinstance(raw, dict) or "id" not in raw:
        raise AppError(
            code="models_yaml_invalid",
            message="models.yaml models entries must be strings or objects with an 'id' field",
            status_code=500,
            extra={"path": path, "bad_item": str(raw)},
        )

    mid = _as_str(raw.get("id"), field="models[].id", path=path)

    backend = (
        _validate_enum(raw.get("backend", defaults["backend"]), field="models[].backend", path=path, allowed=_ALLOWED_BACKENDS)
        or defaults["backend"]
    )
    load_mode = (
        _validate_enum(raw.get("load_mode", defaults["load_mode"]), field="models[].load_mode", path=path, allowed=_ALLOWED_LOAD_MODES)
        or defaults["load_mode"]
    )

    caps_raw = raw.get("capabilities", defaults.get("capabilities"))
    capabilities = _normalize_capabilities(caps_raw, path=path, field="models[].capabilities")

    transformers = _normalize_backend_block(raw.get("transformers", None), path=path, field="models[].transformers")
    llamacpp = _normalize_backend_block(raw.get("llamacpp", None), path=path, field="models[].llamacpp")
    remote = _normalize_backend_block(raw.get("remote", None), path=path, field="models[].remote")

    transformers = _deep_merge_dicts(defaults.get("transformers") or {}, transformers or {})
    llamacpp = _deep_merge_dicts(defaults.get("llamacpp") or {}, llamacpp or {})
    remote = _deep_merge_dicts(defaults.get("remote") or {}, remote or {})

    # back-compat: older flat keys -> fold into remote/transformers if present
    if "remote_base_url" in raw and raw.get("remote_base_url") is not None:
        remote.setdefault("base_url", raw.get("remote_base_url"))
    if "remote_model_id" in raw and raw.get("remote_model_id") is not None:
        remote.setdefault("model_name", raw.get("remote_model_id"))

    notes = _as_opt_str(raw.get("notes"))

    return ModelSpec(
        id=mid,
        backend=backend,  # type: ignore[assignment]
        load_mode=load_mode,  # type: ignore[assignment]
        capabilities=capabilities,
        transformers=transformers or None,
        llamacpp=llamacpp or None,
        remote=remote or None,
        notes=notes,
    )


# -----------------------------
# Model-driven service capability clamp
# -----------------------------


def _cap_bool(caps: Optional[CapabilitiesMap], key: str, default: bool) -> bool:
    if not caps:
        return default
    v = caps.get(key)  # type: ignore[arg-type]
    return bool(v) if isinstance(v, bool) else default


def _apply_effective_service_caps_from_primary(
    *,
    s: Any,
    primary: ModelSpec,
    defaults_caps: Optional[CapabilitiesMap],
) -> Tuple[bool, bool]:
    gen_default = _cap_bool(defaults_caps, "generate", True)
    ex_default = _cap_bool(defaults_caps, "extract", False)

    primary_gen = _cap_bool(primary.capabilities, "generate", gen_default)
    primary_ex = _cap_bool(primary.capabilities, "extract", ex_default)

    settings_gen = bool(getattr(s, "enable_generate", True))
    settings_ex = bool(getattr(s, "enable_extract", True))

    effective_gen = settings_gen and primary_gen
    effective_ex = settings_ex and primary_ex

    try:
        setattr(s, "enable_generate", effective_gen)
        setattr(s, "enable_extract", effective_ex)
    except Exception:
        pass

    os.environ.setdefault("ENABLE_GENERATE", "1" if effective_gen else "0")
    os.environ.setdefault("ENABLE_EXTRACT", "1" if effective_ex else "0")

    return effective_gen, effective_ex


# -----------------------------
# Main loader
# -----------------------------


def load_models_config() -> ModelsConfig:
    s = get_settings()
    path = _resolve_models_yaml_path()

    if path and os.path.exists(path):
        raw = _load_yaml(path)

        # ✅ IMPORTANT: expand ${VAR} before profile selection & normalization
        raw = _expand_env(raw)

        data = _select_profile(raw)

        default_model = data.get("default_model")
        if default_model is not None and not isinstance(default_model, str):
            raise AppError(
                code="models_yaml_invalid",
                message="models.yaml default_model must be a string",
                status_code=500,
                extra={"path": path},
            )

        models_list = data.get("models") or []
        if not isinstance(models_list, list):
            raise AppError(
                code="models_yaml_invalid",
                message="models.yaml models must be a list",
                status_code=500,
                extra={"path": path},
            )

        defaults: Dict[str, Any] = data.get("defaults") or {}
        if defaults and not isinstance(defaults, dict):
            raise AppError(
                code="models_yaml_invalid",
                message="models.yaml defaults must be a mapping (dict) if provided",
                status_code=500,
                extra={"path": path},
            )

        caps_defaults = _normalize_capabilities(defaults.get("capabilities", None), path=path, field="defaults.capabilities")

        norm_defaults: Dict[str, Any] = {
            "backend": _validate_enum(defaults.get("backend", "transformers"), field="defaults.backend", path=path, allowed=_ALLOWED_BACKENDS)
            or "transformers",
            "load_mode": _validate_enum(defaults.get("load_mode", "lazy"), field="defaults.load_mode", path=path, allowed=_ALLOWED_LOAD_MODES)
            or "lazy",
            "capabilities": caps_defaults,
            "transformers": defaults.get("transformers", {}) if isinstance(defaults.get("transformers", {}), dict) else {},
            "llamacpp": defaults.get("llamacpp", {}) if isinstance(defaults.get("llamacpp", {}), dict) else {},
            "remote": defaults.get("remote", {}) if isinstance(defaults.get("remote", {}), dict) else {},
        }

        specs: List[ModelSpec] = []
        for raw_item in models_list:
            specs.append(_normalize_model_entry(raw_item, path=path, defaults=norm_defaults))

        ids = _dedupe_preserve_order([m.id for m in specs])

        if default_model is None:
            if not ids:
                raise AppError(
                    code="models_yaml_invalid",
                    message="models.yaml must define default_model and/or at least one model id in models",
                    status_code=500,
                    extra={"path": path},
                )
            primary_id = ids[0]
        else:
            primary_id = default_model.strip()
            if not primary_id:
                raise AppError(
                    code="models_yaml_invalid",
                    message="models.yaml default_model must be a non-empty string",
                    status_code=500,
                    extra={"path": path},
                )
            if primary_id not in ids:
                ids.insert(0, primary_id)
                specs.insert(
                    0,
                    ModelSpec(
                        id=primary_id,
                        backend=norm_defaults["backend"],
                        load_mode=norm_defaults["load_mode"],
                        capabilities=norm_defaults.get("capabilities"),
                        transformers=dict(norm_defaults.get("transformers") or {}) or None,
                        llamacpp=dict(norm_defaults.get("llamacpp") or {}) or None,
                        remote=dict(norm_defaults.get("remote") or {}) or None,
                        notes="(auto-added because default_model was not listed)",
                    ),
                )

        spec_map: Dict[str, ModelSpec] = {}
        for sp in specs:
            if sp.id not in spec_map:
                spec_map[sp.id] = sp

        ordered_ids = [primary_id] + [x for x in ids if x != primary_id]
        ordered_specs = [spec_map[mid] for mid in ordered_ids if mid in spec_map]

        # legacy wiring support
        try:
            s.model_id = primary_id  # type: ignore[attr-defined]
            s.allowed_models = ordered_ids  # type: ignore[attr-defined]
            s.models_config_path = path  # type: ignore[attr-defined]
        except Exception:
            pass

        try:
            primary_spec = next((m for m in ordered_specs if m.id == primary_id), None)
            if primary_spec is not None:
                _apply_effective_service_caps_from_primary(s=s, primary=primary_spec, defaults_caps=caps_defaults)
        except Exception:
            pass

        logger.info(
            "models: loaded path=%s requested_profile=%r source=%r used_profile=%r primary_id=%s model_ids=%d",
            path,
            data.get("_selected_profile_requested", None),
            data.get("_selected_profile_source", None),
            data.get("_selected_profile_used", None),
            primary_id,
            len(ordered_ids),
        )

        selected = data.get("_selected_profile_used", None)
        return ModelsConfig(
            primary_id=str(primary_id),
            model_ids=[str(x) for x in ordered_ids],
            models=ordered_specs,
            defaults={"path": path, "selected_profile": selected, **norm_defaults},
        )

    # Fall back to Settings (legacy)
    primary_id = getattr(s, "model_id", None)
    model_ids = list(getattr(s, "all_model_ids", []) or [])
    if not primary_id or not isinstance(primary_id, str) or not primary_id.strip():
        raise AppError(
            code="model_config_invalid",
            message="Primary model id is missing or invalid",
            status_code=500,
            extra={"primary_id": str(primary_id)},
        )

    model_ids = [str(x) for x in model_ids if str(x).strip()]
    model_ids = _dedupe_preserve_order(model_ids)
    if primary_id not in model_ids:
        model_ids.insert(0, primary_id)

    try:
        s.model_id = primary_id  # type: ignore[attr-defined]
        s.allowed_models = model_ids  # type: ignore[attr-defined]
    except Exception:
        pass

    specs = [
        ModelSpec(
            id=mid,
            backend="transformers",
            load_mode="lazy" if mid != primary_id else "eager",
            capabilities=None,
            transformers=None,
            llamacpp=None,
            remote=None,
            notes="(from settings)",
        )
        for mid in model_ids
    ]

    return ModelsConfig(primary_id=str(primary_id), model_ids=[str(x) for x in model_ids], models=specs, defaults={"path": None})