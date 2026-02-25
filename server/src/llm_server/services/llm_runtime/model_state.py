# server/src/llm_server/services/llm_runtime/model_state.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class ModelStateSnapshot:
    """
    Structured runtime state for the model layer.

    Design goals:
      - Centralize and standardize what "model state" means.
      - Avoid scattering app.state flags across the codebase.
      - Provide a stable snapshot view for health/admin/debug.

    IMPORTANT:
      - The canonical storage is app.state.model_state (this object).
      - For backward compatibility, we also mirror into legacy app.state fields:
          - app.state.model_error
          - app.state.model_load_mode
          - app.state.model_loaded
          - app.state.loaded_model_id
          - app.state.runtime_default_model_id
          - app.state.models_config
          - app.state.llm
    """

    model_error: Optional[str]
    model_load_mode: str
    model_loaded: bool
    loaded_model_id: Optional[str]
    runtime_default_model_id: Optional[str]

    models_config_loaded: bool
    registry_kind: Optional[str]


@dataclass
class ModelStateMutable:
    """
    Mutable storage attached to app.state.model_state.

    NOTE:
      - Keep it intentionally small. This is state, not business logic.
      - Business logic belongs in llm_loader / admin / deps, etc.
    """

    model_error: Optional[str] = None
    model_load_mode: str = "lazy"
    model_loaded: bool = False
    loaded_model_id: Optional[str] = None
    runtime_default_model_id: Optional[str] = None

    def snapshot(self, *, llm: Any = None, models_config: Any = None) -> ModelStateSnapshot:
        reg_kind = type(llm).__name__ if llm is not None else None
        return ModelStateSnapshot(
            model_error=self.model_error,
            model_load_mode=self.model_load_mode,
            model_loaded=bool(self.model_loaded),
            loaded_model_id=self.loaded_model_id,
            runtime_default_model_id=self.runtime_default_model_id,
            models_config_loaded=bool(models_config is not None),
            registry_kind=reg_kind,
        )


class ModelStateStore:
    """
    Read/write adapter over app.state.

    This is the ONLY place that should know:
      - where the canonical state lives (app.state.model_state)
      - which legacy fields must be mirrored for compatibility
    """

    def __init__(self, app_state: Any) -> None:
        self._s = app_state
        self._ensure_exists()

    def _ensure_exists(self) -> None:
        ms = getattr(self._s, "model_state", None)
        if not isinstance(ms, ModelStateMutable):
            ms = ModelStateMutable()
            setattr(self._s, "model_state", ms)

        # Bring forward legacy fields into canonical object if present
        legacy_error = getattr(self._s, "model_error", None)
        if isinstance(legacy_error, str) and legacy_error.strip():
            ms.model_error = legacy_error.strip()

        legacy_mode = getattr(self._s, "model_load_mode", None)
        if isinstance(legacy_mode, str) and legacy_mode.strip():
            ms.model_load_mode = legacy_mode.strip().lower()

        ms.model_loaded = bool(getattr(self._s, "model_loaded", ms.model_loaded))

        legacy_loaded_id = getattr(self._s, "loaded_model_id", None)
        if isinstance(legacy_loaded_id, str) and legacy_loaded_id.strip():
            ms.loaded_model_id = legacy_loaded_id.strip()

        legacy_runtime_default = getattr(self._s, "runtime_default_model_id", None)
        if isinstance(legacy_runtime_default, str) and legacy_runtime_default.strip():
            ms.runtime_default_model_id = legacy_runtime_default.strip()

        # Mirror canonical -> legacy once so callers get consistent values
        self._mirror_to_legacy(ms)

    def _mirror_to_legacy(self, ms: ModelStateMutable) -> None:
        setattr(self._s, "model_error", ms.model_error)
        setattr(self._s, "model_load_mode", ms.model_load_mode)
        setattr(self._s, "model_loaded", bool(ms.model_loaded))
        setattr(self._s, "loaded_model_id", ms.loaded_model_id)
        setattr(self._s, "runtime_default_model_id", ms.runtime_default_model_id)

    def get_mut(self) -> ModelStateMutable:
        self._ensure_exists()
        ms = getattr(self._s, "model_state")
        return ms  # type: ignore[return-value]

    def snapshot(self) -> ModelStateSnapshot:
        ms = self.get_mut()
        llm = getattr(self._s, "llm", None)
        cfg = getattr(self._s, "models_config", None)
        return ms.snapshot(llm=llm, models_config=cfg)

    # ---- setters (canonical + mirror) ----

    def set_model_error(self, err: Optional[str]) -> None:
        ms = self.get_mut()
        ms.model_error = (err.strip() if isinstance(err, str) and err.strip() else None)
        self._mirror_to_legacy(ms)

    def set_model_load_mode(self, mode: str) -> None:
        ms = self.get_mut()
        m = (mode or "").strip().lower()
        ms.model_load_mode = m or "lazy"
        self._mirror_to_legacy(ms)

    def set_model_loaded(self, loaded: bool) -> None:
        ms = self.get_mut()
        ms.model_loaded = bool(loaded)
        self._mirror_to_legacy(ms)

    def set_loaded_model_id(self, model_id: Optional[str]) -> None:
        ms = self.get_mut()
        ms.loaded_model_id = (model_id.strip() if isinstance(model_id, str) and model_id.strip() else None)
        self._mirror_to_legacy(ms)

    def set_runtime_default_model_id(self, model_id: Optional[str]) -> None:
        ms = self.get_mut()
        ms.runtime_default_model_id = (model_id.strip() if isinstance(model_id, str) and model_id.strip() else None)
        self._mirror_to_legacy(ms)