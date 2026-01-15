# src/llm_server/services/llm_config.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

import yaml

from llm_server.core.config import settings
from llm_server.core.errors import AppError


# -----------------------------
# Types / allowed values
# -----------------------------

Backend = Literal["local", "remote"]
LoadMode = Literal["eager", "lazy", "off"]
Device = Literal["auto", "cuda", "mps", "cpu"]
DType = Literal["float16", "bfloat16", "float32"]

# Keep this open-ended, but validate known values.
Quantization = Optional[str]  # e.g. "int8", "int4", "nf4", None


_ALLOWED_BACKENDS = {"local", "remote"}
_ALLOWED_LOAD_MODES = {"eager", "lazy", "off"}
_ALLOWED_DEVICES = {"auto", "cuda", "mps", "cpu"}
_ALLOWED_DTYPES = {"float16", "bfloat16", "float32"}
_ALLOWED_QUANT = {None, "int8", "int4", "nf4"}  # extend later as you add support


# -----------------------------
# Normalized config objects
# -----------------------------

@dataclass(frozen=True)
class ModelSpec:
    """
    Normalized single-model spec.
    """
    id: str
    backend: Backend = "local"
    load_mode: LoadMode = "lazy"
    dtype: Optional[DType] = None
    device: Device = "auto"
    text_only: Optional[bool] = None
    max_context: Optional[int] = None
    trust_remote_code: bool = False
    quantization: Quantization = None
    notes: Optional[str] = None


@dataclass(frozen=True)
class ModelsConfig:
    """
    Normalized model configuration for the service.

    primary_id: the default model id
    model_ids:  ordered unique model ids, primary first
    models:     list of ModelSpec in same order as model_ids
    defaults:   any derived defaults used during normalization (for debugging)
    """
    primary_id: str
    model_ids: List[str]
    models: List[ModelSpec]
    defaults: Dict[str, Any]


# -----------------------------
# Helpers
# -----------------------------

def _models_yaml_path() -> str:
    return str(getattr(settings, "models_config_path", None) or "models.yaml")


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for x in items:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


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


def _as_opt_int(x: Any, *, field: str, path: str) -> Optional[int]:
    if x is None:
        return None
    if isinstance(x, bool):
        # bool is int subclass; reject it explicitly
        raise AppError(
            code="models_yaml_invalid",
            message=f"models.yaml {field} must be an integer",
            status_code=500,
            extra={"path": path, "field": field, "value": x},
        )
    if isinstance(x, int):
        return x
    raise AppError(
        code="models_yaml_invalid",
        message=f"models.yaml {field} must be an integer",
        status_code=500,
        extra={"path": path, "field": field, "value": x},
    )


def _as_opt_bool(x: Any, *, field: str, path: str) -> Optional[bool]:
    if x is None:
        return None
    if isinstance(x, bool):
        return x
    raise AppError(
        code="models_yaml_invalid",
        message=f"models.yaml {field} must be a boolean",
        status_code=500,
        extra={"path": path, "field": field, "value": x},
    )


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
        with open(path, "r") as f:
            data = yaml.safe_load(f)
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


def _normalize_model_entry(
    raw: Any,
    *,
    path: str,
    defaults: Dict[str, Any],
) -> ModelSpec:
    """
    Accepts:
      - "model_id" (string)
      - {"id": "..."} (minimal)
      - {"id": "...", backend/load_mode/...} (full)
    """
    if isinstance(raw, str):
        mid = _as_str(raw, field="models[]", path=path)
        return ModelSpec(
            id=mid,
            backend=defaults["backend"],
            load_mode=defaults["load_mode"],
            dtype=defaults["dtype"],
            device=defaults["device"],
            text_only=defaults["text_only"],
            max_context=defaults["max_context"],
            trust_remote_code=defaults["trust_remote_code"],
            quantization=defaults["quantization"],
            notes=None,
        )

    if not isinstance(raw, dict):
        raise AppError(
            code="models_yaml_invalid",
            message="models.yaml models entries must be strings or objects with an 'id' field",
            status_code=500,
            extra={"path": path, "bad_item": str(raw)},
        )

    if "id" not in raw:
        raise AppError(
            code="models_yaml_invalid",
            message="models.yaml models object entries must contain 'id'",
            status_code=500,
            extra={"path": path, "bad_item": str(raw)},
        )

    mid = _as_str(raw.get("id"), field="models[].id", path=path)

    backend = _validate_enum(
        raw.get("backend", defaults["backend"]),
        field="models[].backend",
        path=path,
        allowed=_ALLOWED_BACKENDS,
    ) or defaults["backend"]

    load_mode = _validate_enum(
        raw.get("load_mode", defaults["load_mode"]),
        field="models[].load_mode",
        path=path,
        allowed=_ALLOWED_LOAD_MODES,
    ) or defaults["load_mode"]

    device = _validate_enum(
        raw.get("device", defaults["device"]),
        field="models[].device",
        path=path,
        allowed=_ALLOWED_DEVICES,
    ) or defaults["device"]

    dtype_raw = raw.get("dtype", defaults["dtype"])
    dtype = None
    if dtype_raw is not None:
        dtype = _validate_enum(
            dtype_raw,
            field="models[].dtype",
            path=path,
            allowed=_ALLOWED_DTYPES,
        )

    quant = raw.get("quantization", defaults["quantization"])
    if isinstance(quant, str):
        quant = quant.strip().lower()
    if quant not in _ALLOWED_QUANT:
        raise AppError(
            code="models_yaml_invalid",
            message="models.yaml models[].quantization has invalid value",
            status_code=500,
            extra={"path": path, "field": "models[].quantization", "value": quant, "allowed": sorted([x for x in _ALLOWED_QUANT if x is not None]) + [None]},
        )

    text_only = _as_opt_bool(raw.get("text_only", defaults["text_only"]), field="models[].text_only", path=path)
    max_context = _as_opt_int(raw.get("max_context", defaults["max_context"]), field="models[].max_context", path=path)

    trc = raw.get("trust_remote_code", defaults["trust_remote_code"])
    if trc is None:
        trc = defaults["trust_remote_code"]
    if not isinstance(trc, bool):
        raise AppError(
            code="models_yaml_invalid",
            message="models.yaml models[].trust_remote_code must be a boolean",
            status_code=500,
            extra={"path": path, "field": "models[].trust_remote_code", "value": trc},
        )

    notes = _as_opt_str(raw.get("notes"))

    return ModelSpec(
        id=mid,
        backend=backend,               # type: ignore[assignment]
        load_mode=load_mode,           # type: ignore[assignment]
        dtype=dtype,                   # type: ignore[assignment]
        device=device,                 # type: ignore[assignment]
        text_only=text_only,
        max_context=max_context,
        trust_remote_code=bool(trc),
        quantization=quant,            # type: ignore[assignment]
        notes=notes,
    )


def load_models_config() -> ModelsConfig:
    """
    Load model specs from models.yaml if present, otherwise fall back to Settings.

    Also updates settings.model_id and settings.allowed_models best-effort
    to keep the rest of the app consistent.
    """
    path = _models_yaml_path()
    if path and os.path.exists(path):
        data = _load_yaml(path)

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

        # Top-level defaults (optional). These apply unless overridden per model.
        defaults: Dict[str, Any] = data.get("defaults") or {}
        if defaults and not isinstance(defaults, dict):
            raise AppError(
                code="models_yaml_invalid",
                message="models.yaml defaults must be a mapping (dict) if provided",
                status_code=500,
                extra={"path": path},
            )

        norm_defaults: Dict[str, Any] = {
            "backend": _validate_enum(defaults.get("backend", "local"), field="defaults.backend", path=path, allowed=_ALLOWED_BACKENDS) or "local",
            "load_mode": _validate_enum(defaults.get("load_mode", "lazy"), field="defaults.load_mode", path=path, allowed=_ALLOWED_LOAD_MODES) or "lazy",
            "device": _validate_enum(defaults.get("device", "auto"), field="defaults.device", path=path, allowed=_ALLOWED_DEVICES) or "auto",
            "dtype": None,
            "text_only": defaults.get("text_only", None),
            "max_context": defaults.get("max_context", None),
            "trust_remote_code": bool(defaults.get("trust_remote_code", False)),
            "quantization": defaults.get("quantization", None),
        }

        if norm_defaults["text_only"] is not None and not isinstance(norm_defaults["text_only"], bool):
            raise AppError(
                code="models_yaml_invalid",
                message="models.yaml defaults.text_only must be a boolean",
                status_code=500,
                extra={"path": path, "field": "defaults.text_only", "value": norm_defaults["text_only"]},
            )

        if norm_defaults["max_context"] is not None and not isinstance(norm_defaults["max_context"], int):
            raise AppError(
                code="models_yaml_invalid",
                message="models.yaml defaults.max_context must be an integer",
                status_code=500,
                extra={"path": path, "field": "defaults.max_context", "value": norm_defaults["max_context"]},
            )

        dtype_default = defaults.get("dtype", None)
        if dtype_default is not None:
            norm_defaults["dtype"] = _validate_enum(dtype_default, field="defaults.dtype", path=path, allowed=_ALLOWED_DTYPES)

        quant_default = defaults.get("quantization", None)
        if isinstance(quant_default, str):
            quant_default = quant_default.strip().lower()
        if quant_default not in _ALLOWED_QUANT:
            raise AppError(
                code="models_yaml_invalid",
                message="models.yaml defaults.quantization has invalid value",
                status_code=500,
                extra={"path": path, "field": "defaults.quantization", "value": quant_default, "allowed": sorted([x for x in _ALLOWED_QUANT if x is not None]) + [None]},
            )
        norm_defaults["quantization"] = quant_default

        # Normalize model entries
        specs: List[ModelSpec] = []
        for raw in models_list:
            specs.append(_normalize_model_entry(raw, path=path, defaults=norm_defaults))

        ids = [m.id for m in specs]
        ids = _dedupe_preserve_order(ids)

        # Determine primary/default id
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
            if primary_id not in ids:
                # default must be included in models list for consistency
                ids.insert(0, primary_id)
                specs.insert(
                    0,
                    ModelSpec(
                        id=primary_id,
                        backend=norm_defaults["backend"],
                        load_mode=norm_defaults["load_mode"],
                        dtype=norm_defaults["dtype"],
                        device=norm_defaults["device"],
                        text_only=norm_defaults["text_only"],
                        max_context=norm_defaults["max_context"],
                        trust_remote_code=norm_defaults["trust_remote_code"],
                        quantization=norm_defaults["quantization"],
                        notes="(auto-added because default_model was not listed)",
                    ),
                )

        # Reorder so primary first (and keep matching specs order)
        # Build map from id -> first spec occurrence
        spec_map: Dict[str, ModelSpec] = {}
        for s in specs:
            if s.id not in spec_map:
                spec_map[s.id] = s

        ordered_ids = [primary_id] + [x for x in ids if x != primary_id]
        ordered_specs = [spec_map[mid] for mid in ordered_ids if mid in spec_map]

        # Best-effort update settings for compatibility
        try:
            settings.model_id = primary_id  # type: ignore[attr-defined]
            settings.allowed_models = ordered_ids  # type: ignore[attr-defined]
        except Exception:
            pass

        return ModelsConfig(
            primary_id=str(primary_id),
            model_ids=[str(x) for x in ordered_ids],
            models=ordered_specs,
            defaults={"path": path, **norm_defaults},
        )

    # Fallback to Settings (legacy)
    primary_id = settings.model_id
    model_ids = settings.all_model_ids

    if not primary_id or not isinstance(primary_id, str):
        raise AppError(
            code="model_config_invalid",
            message="Primary model id is missing or invalid",
            status_code=500,
            extra={"primary_id": str(primary_id)},
        )

    model_ids = [str(x) for x in (model_ids or []) if str(x).strip()]
    model_ids = _dedupe_preserve_order(model_ids)
    if primary_id not in model_ids:
        model_ids.insert(0, primary_id)

    try:
        settings.model_id = primary_id  # type: ignore[attr-defined]
        settings.allowed_models = model_ids  # type: ignore[attr-defined]
    except Exception:
        pass

    # Create minimal specs from Settings defaults
    specs = [
        ModelSpec(
            id=mid,
            backend="local",
            load_mode="lazy" if mid != primary_id else "eager",
            dtype=None,
            device="auto",
            text_only=None,
            max_context=None,
            trust_remote_code=False,
            quantization=None,
            notes="(from settings)",
        )
        for mid in model_ids
    ]

    return ModelsConfig(primary_id=str(primary_id), model_ids=[str(x) for x in model_ids], models=specs, defaults={"path": None})