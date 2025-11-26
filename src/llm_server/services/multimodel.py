from __future__ import annotations
from typing import Dict, Optional, Type

class MultiModelManager:
    def __init__(
        self,
        managers: Dict[str, object],
        default_model_id: str,
    ):
        self.managers = managers
        self.default_model_id = default_model_id

    # ---------- factories ----------

    @classmethod
    def from_settings(cls, settings, base_manager_cls: Type):
        model_ids = list(set(
            [settings.model_id] + (settings.allowed_models or [])
        ))

        managers = {}

        for model_id in model_ids:
            mgr = base_manager_cls.from_settings(settings)
            mgr.model_id = model_id
            managers[model_id] = mgr

        return cls(
            managers=managers,
            default_model_id=settings.model_id,
        )

    # ---------- helpers ----------

    def list_models(self) -> list[str]:
        return sorted(self.managers.keys())

    def get_manager(self, model: Optional[str]):
        if model is None:
            return self.managers[self.default_model_id]

        if model not in self.managers:
            raise ValueError(
                f"Model '{model}' not allowed. Allowed: {self.list_models()}"
            )

        return self.managers[model]

    # ---------- public API (mirrors ModelManager) ----------

    def ensure_loaded(self):
        # Load default on startup only
        self.managers[self.default_model_id].ensure_loaded()

    def generate(self, *, model=None, **kwargs):
        mgr = self.get_manager(model)
        return mgr.generate(**kwargs)

    def stream(self, *, model=None, **kwargs):
        mgr = self.get_manager(model)
        return mgr.stream(**kwargs)