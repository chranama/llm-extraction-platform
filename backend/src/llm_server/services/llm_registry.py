# src/llm_server/services/llm_registry.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from llm_server.core.errors import AppError


@dataclass
class ModelStatus:
    """
    Lightweight status view used for /models and readiness/debug.
    """
    model_id: str
    backend: str
    load_mode: str
    loaded: Optional[bool]  # None = unknown
    detail: Optional[str] = None


class MultiModelManager:
    """
    Registry / router for multiple model backends.

    Contract:
      - __getitem__(model_id) -> backend object
      - __contains__(model_id) -> bool
      - list_models() -> [ids]
      - ensure_loaded(): loads ONLY default model (cloud-friendly)
      - load_all(): loads all models (admin/manual)
      - is_loaded(): best-effort status of default model
      - is_loaded_model(model_id): best-effort status for any model
      - ensure_loaded_model(model_id): load a specific model (respects existence)
      - status(): list[ModelStatus] for observability/UI
    """

    def __init__(
        self,
        models: Dict[str, Any],
        default_id: str,
        model_meta: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        self._models = models
        self.default_id = default_id
        # Optional: registry metadata (backend/load_mode/etc.) provided by llm.py/llm_factory
        self._meta: Dict[str, Dict[str, Any]] = model_meta or {}

    # --------------------
    # Introspection
    # --------------------

    @property
    def models(self) -> Dict[str, Any]:
        return self._models

    def list_models(self) -> List[str]:
        return list(self._models.keys())

    def _missing(self, model_id: str) -> AppError:
        return AppError(
            code="model_missing",
            message=f"Model '{model_id}' not found in LLM registry",
            status_code=500,
            extra={"model_id": model_id, "available": self.list_models(), "default_id": self.default_id},
        )

    def get(self, model_id: str) -> Any:
        if model_id not in self._models:
            raise self._missing(model_id)
        return self._models[model_id]

    def __getitem__(self, model_id: str) -> Any:
        return self.get(model_id)

    def __contains__(self, model_id: object) -> bool:
        return bool(model_id in self._models)

    # --------------------
    # Loading controls
    # --------------------

    def ensure_loaded(self) -> None:
        """
        Cloud-friendly default: load ONLY the default model.
        """
        self.ensure_loaded_model(self.default_id)

    def ensure_loaded_model(self, model_id: str) -> None:
        mgr = self._models.get(model_id)
        if mgr is None:
            raise self._missing(model_id)

        fn = getattr(mgr, "ensure_loaded", None)
        if callable(fn):
            fn()

    def load_all(self) -> None:
        """
        Admin/manual: loads all models that support ensure_loaded().
        """
        for mgr in self._models.values():
            fn = getattr(mgr, "ensure_loaded", None)
            if callable(fn):
                fn()

    # --------------------
    # Readiness / status
    # --------------------

    def is_loaded(self) -> bool:
        """
        Back-compat: status of default model.
        """
        return self.is_loaded_model(self.default_id)

    def is_loaded_model(self, model_id: str) -> bool:
        mgr = self._models.get(model_id)
        if mgr is None:
            return False

        fn = getattr(mgr, "is_loaded", None)
        if callable(fn):
            try:
                return bool(fn())
            except Exception:
                return False

        # Heuristic: underlying handles exist for local HF managers
        m = getattr(mgr, "_model", None)
        t = getattr(mgr, "_tokenizer", None)
        if (m is not None) and (t is not None):
            return True

        # Remote clients and other backends might not expose load state
        # If it has ensure_loaded() but no state, treat as unknown -> False
        return False

    def status(self) -> List[ModelStatus]:
        """
        Returns a stable, UI-friendly status list for all models.
        Uses registry metadata if provided, otherwise best-effort.
        """
        out: List[ModelStatus] = []
        for mid in self.list_models():
            meta = self._meta.get(mid, {})
            backend = str(meta.get("backend") or type(self._models[mid]).__name__)
            load_mode = str(meta.get("load_mode") or "unknown")

            loaded: Optional[bool]
            try:
                loaded = bool(self.is_loaded_model(mid))
            except Exception:
                loaded = None

            detail = None
            if mid == self.default_id:
                detail = "default"

            out.append(
                ModelStatus(
                    model_id=mid,
                    backend=backend,
                    load_mode=load_mode,
                    loaded=loaded,
                    detail=detail,
                )
            )
        return out