# src/llm_server/services/llm_runtime/llm_registry.py
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
    capabilities: Optional[List[str]] = None
    readiness_mode: Optional[str] = None


class MultiModelManager:
    """
    Registry / router for multiple model backends.

    Contract:
      - __getitem__(model_id) -> backend object
      - __contains__(model_id) -> bool
      - list_models() -> [ids]
      - default() -> backend for default model
      - ensure_loaded(): loads ONLY default model (cloud-friendly)
      - load_all(): loads all models (admin/manual)
      - is_loaded(): best-effort status of default model
      - is_loaded_model(model_id): best-effort status for any model
      - ensure_loaded_model(model_id): load a specific model (respects existence)
      - status(): list[ModelStatus] for observability/UI

    Capability-aware:
      - has_capability(model_id, cap) -> bool
      - require_capability(model_id, cap) -> None (raises AppError)
      - models_for_capability(cap) -> list[str]
      - default_for_capability(cap) -> str (best-effort; falls back to default_id)

    Capability meta formats supported (model_meta["capabilities"]):
      - None: unspecified => allow all (fail-open)
      - dict: {"generate": True, "extract": False} (missing key => True)
      - list/tuple/set: ["generate", "extract"] (allowed set)
      - str: "generate" (single cap)
    """

    def __init__(
        self,
        models: Dict[str, Any],
        default_id: str,
        model_meta: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        self._models = models
        self.default_id = default_id
        self._meta: Dict[str, Dict[str, Any]] = model_meta or {}

    @property
    def models(self) -> Dict[str, Any]:
        return self._models

    def list_models(self) -> List[str]:
        return list(self._models.keys())

    def default(self) -> Any:
        return self.get(self.default_id)

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
    # Capability gating
    # --------------------

    def _capabilities_meta(self, model_id: str) -> object | None:
        meta = self._meta.get(model_id, {}) or {}
        return meta.get("capabilities", None)

    def has_capability(self, model_id: str, capability: str) -> bool:
        if model_id not in self._models:
            return False

        cap = (capability or "").strip().lower()
        if not cap:
            return True

        caps_meta = self._capabilities_meta(model_id)

        if caps_meta is None:
            return True

        if isinstance(caps_meta, dict):
            v = caps_meta.get(cap)
            return True if v is None else bool(v)

        if isinstance(caps_meta, str):
            s = caps_meta.strip().lower()
            return True if not s else (cap == s)

        if isinstance(caps_meta, (list, tuple, set)):
            allowed = {str(x).strip().lower() for x in caps_meta if isinstance(x, str) and str(x).strip()}
            return cap in allowed

        return True

    def require_capability(self, model_id: str, capability: str) -> None:
        if model_id not in self._models:
            raise self._missing(model_id)

        cap = (capability or "").strip().lower()
        if not cap:
            return

        if not self.has_capability(model_id, cap):
            caps_meta = self._capabilities_meta(model_id)
            raise AppError(
                code="capability_not_supported",
                message=f"Model '{model_id}' does not support capability '{cap}'.",
                status_code=400,
                extra={
                    "model_id": model_id,
                    "capability": cap,
                    "model_capabilities": caps_meta,
                    "available_models": self.list_models(),
                },
            )

    def models_for_capability(self, capability: str) -> List[str]:
        cap = (capability or "").strip().lower()
        if not cap:
            return self.list_models()
        return [mid for mid in self.list_models() if self.has_capability(mid, cap)]

    def default_for_capability(self, capability: str) -> str:
        cap = (capability or "").strip().lower()
        if not cap:
            return self.default_id

        if self.has_capability(self.default_id, cap):
            return self.default_id

        candidates = self.models_for_capability(cap)
        return candidates[0] if candidates else self.default_id

    def _cap_list_for_status(self, model_id: str) -> Optional[List[str]]:
        caps_meta = self._capabilities_meta(model_id)
        if caps_meta is None:
            return None

        order = {"generate": 0, "extract": 1}

        if isinstance(caps_meta, dict):
            out: List[str] = []
            for k, v in caps_meta.items():
                if not isinstance(k, str):
                    continue
                kk = k.strip().lower()
                if kk and bool(v):
                    out.append(kk)
            out = sorted(set(out), key=lambda x: order.get(x, 999))
            return out or None

        if isinstance(caps_meta, str):
            s = caps_meta.strip().lower()
            return [s] if s else None

        if isinstance(caps_meta, (list, tuple, set)):
            raw: List[str] = []
            for x in caps_meta:
                if not isinstance(x, str):
                    continue
                s = x.strip().lower()
                if s:
                    raw.append(s)
            # de-dupe preserving order, then sort by preferred order
            seen: set[str] = set()
            out: List[str] = []
            for x in raw:
                if x not in seen:
                    out.append(x)
                    seen.add(x)
            out.sort(key=lambda x: order.get(x, 999))
            return out or None

        return None

    # --------------------
    # Loading controls
    # --------------------

    def ensure_loaded(self) -> None:
        self.ensure_loaded_model(self.default_id)

    def ensure_loaded_model(self, model_id: str) -> None:
        mgr = self._models.get(model_id)
        if mgr is None:
            raise self._missing(model_id)

        fn = getattr(mgr, "ensure_loaded", None)
        if callable(fn):
            fn()

    def load_all(self) -> None:
        for mgr in self._models.values():
            fn = getattr(mgr, "ensure_loaded", None)
            if callable(fn):
                fn()

    # --------------------
    # Readiness / status
    # --------------------

    def is_loaded(self) -> bool:
        return self.is_loaded_model(self.default_id)

    def is_loaded_model(self, model_id: str) -> bool:
        mgr = self._models.get(model_id)
        if mgr is None:
            return False

        fn = getattr(mgr, "is_loaded", None)
        if callable(fn):
            try:
                return bool(fn())
            except AttributeError:
                pass
            except Exception:
                pass

        m = getattr(mgr, "_model", None)
        t = getattr(mgr, "_tokenizer", None)
        if (m is not None) and (t is not None):
            return True

        return False

    def _readiness_mode_for_status(self, model_id: str) -> Optional[str]:
        meta = self._meta.get(model_id, {}) or {}
        v = meta.get("readiness_mode", None)
        if isinstance(v, str) and v.strip():
            return v.strip().lower()
        return None

    def status(self) -> List[ModelStatus]:
        out: List[ModelStatus] = []
        for mid in self.list_models():
            meta = self._meta.get(mid, {}) or {}
            backend = str(meta.get("backend") or type(self._models[mid]).__name__)
            load_mode = str(meta.get("load_mode") or "unknown")

            loaded: Optional[bool]
            try:
                loaded = bool(self.is_loaded_model(mid))
            except Exception:
                loaded = None

            detail = "default" if mid == self.default_id else None
            caps = self._cap_list_for_status(mid)
            readiness_mode = self._readiness_mode_for_status(mid)

            out.append(
                ModelStatus(
                    model_id=mid,
                    backend=backend,
                    load_mode=load_mode,
                    loaded=loaded,
                    detail=detail,
                    capabilities=caps,
                    readiness_mode=readiness_mode,
                )
            )
        return out