# server/src/llm_server/services/llm_runtime/llm_loader.py
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import anyio

from llm_server.core.errors import AppError
from llm_server.services.llm_runtime.llm_build import build_llm_from_settings
from llm_server.services.llm_runtime.llm_config import ModelsConfig, load_models_config
from llm_server.services.llm_runtime.llm_registry import MultiModelManager
from llm_server.services.llm_runtime.model_state import ModelStateStore
from llm_server.services.llm_runtime.metrics import (
    LLM_LOADER_OPS_FAIL_TOTAL,
    LLM_LOADER_OPS_TOTAL,
    LLM_LOADER_OP_LATENCY_SECONDS,
    set_state_gauges,
)

logger = logging.getLogger("llm_server.llm_loader")


@dataclass(frozen=True)
class LoadResult:
    model_id: str
    loaded: bool  # True only when in-process weights are loaded (transformers)
    load_mode: str
    detail: Dict[str, Any]


@dataclass(frozen=True)
class ProbeResult:
    model_id: str
    ok: bool
    detail: Dict[str, Any]


class RuntimeModelLoader:
    """
    Runtime model loading + registry control plane.

    Canonical state:
      - stored in app.state.model_state (ModelStateStore)
      - mirrored into legacy fields for compatibility

    app.state also holds:
      - app.state.llm
      - app.state.models_config
    """

    def __init__(self, app_state: Any) -> None:
        self._state = app_state
        self._lock = anyio.Lock()
        self._ms = ModelStateStore(app_state)

    # ----------------------------
    # Internal helpers
    # ----------------------------

    def _get_llm(self) -> Any:
        return getattr(self._state, "llm", None)

    def _set_llm(self, llm: Any) -> None:
        setattr(self._state, "llm", llm)

    def _set_models_config(self, cfg: ModelsConfig) -> None:
        setattr(self._state, "models_config", cfg)

    def _get_models_config(self) -> Optional[ModelsConfig]:
        cfg = getattr(self._state, "models_config", None)
        return cfg if cfg is not None else None

    def _get_model_load_mode(self) -> str:
        snap = self._ms.snapshot()
        m = (snap.model_load_mode or "").strip().lower()
        return m or "lazy"

    async def _ensure_llm_exists(self) -> Any:
        llm = self._get_llm()
        if llm is None:
            llm = build_llm_from_settings()
            self._set_llm(llm)
        return llm

    @staticmethod
    def _detect_backend_name(backend: Any) -> Optional[str]:
        try:
            v = getattr(backend, "backend_name", None)
            if isinstance(v, str) and v.strip():
                return v.strip().lower()
        except Exception:
            pass
        return None

    @staticmethod
    def _ensure_loaded_sync(backend: Any) -> None:
        """
        Call backend.ensure_loaded() if present.
        This is the only place we might trigger weight loading.
        """
        fn = getattr(backend, "ensure_loaded", None)
        if callable(fn):
            fn()

    async def _ensure_loaded_async(self, backend: Any) -> None:
        await anyio.to_thread.run_sync(self._ensure_loaded_sync, backend)

    def _resolve_backend(self, llm: Any, model_id: str) -> Tuple[str, Any]:
        if isinstance(llm, MultiModelManager):
            if model_id not in llm:
                raise AppError(
                    code="model_missing",
                    message=f"Model '{model_id}' not found in LLM registry",
                    status_code=400,
                    extra={"available": llm.list_models(), "default_id": llm.default_id},
                )
            return model_id, llm[model_id]

        resolved = model_id or getattr(llm, "model_id", None) or "default"
        return str(resolved), llm

    def _emit_state_gauges(self) -> None:
        snap = self._ms.snapshot()
        set_state_gauges(
            model_loaded=bool(snap.model_loaded),
            model_error=bool(snap.model_error),
            loaded_model_id=snap.loaded_model_id,
            runtime_default_model_id=snap.runtime_default_model_id,
        )

    def _metric_inc(self, counter, **labels) -> None:
        try:
            if counter is not None:
                counter.labels(**labels).inc()
        except Exception:
            pass

    def _metric_obs(self, hist, value: float, **labels) -> None:
        try:
            if hist is not None:
                hist.labels(**labels).observe(value)
        except Exception:
            pass

    # ----------------------------
    # Public API
    # ----------------------------

    async def refresh_models_config(self) -> ModelsConfig:
        op = "refresh_models_config"
        t0 = time.perf_counter()
        self._metric_inc(LLM_LOADER_OPS_TOTAL, op=op)

        try:
            async with self._lock:
                cfg = load_models_config()
                self._set_models_config(cfg)
                self._emit_state_gauges()
                return cfg
        except Exception as e:
            self._metric_inc(LLM_LOADER_OPS_FAIL_TOTAL, op=op, reason=type(e).__name__)
            raise
        finally:
            self._metric_obs(LLM_LOADER_OP_LATENCY_SECONDS, time.perf_counter() - t0, op=op)

    async def rebuild_llm_registry(self) -> Any:
        op = "rebuild_llm_registry"
        t0 = time.perf_counter()
        self._metric_inc(LLM_LOADER_OPS_TOTAL, op=op)

        try:
            async with self._lock:
                llm = build_llm_from_settings()
                self._set_llm(llm)
                self._emit_state_gauges()
                return llm
        except Exception as e:
            self._metric_inc(LLM_LOADER_OPS_FAIL_TOTAL, op=op, reason=type(e).__name__)
            raise
        finally:
            self._metric_obs(LLM_LOADER_OP_LATENCY_SECONDS, time.perf_counter() - t0, op=op)

    async def load_model(self, model_id: str, *, force: bool = False) -> LoadResult:
        op = "load_model"
        t0 = time.perf_counter()
        self._metric_inc(LLM_LOADER_OPS_TOTAL, op=op)

        mid = (model_id or "").strip()
        if not mid:
            self._metric_inc(LLM_LOADER_OPS_FAIL_TOTAL, op=op, reason="invalid_request")
            raise AppError(code="invalid_request", message="model_id is required", status_code=400)

        async with self._lock:
            # clear previous error
            self._ms.set_model_error(None)

            llm = await self._ensure_llm_exists()
            resolved_id, backend = self._resolve_backend(llm, mid)

            mode = self._get_model_load_mode()
            detail: Dict[str, Any] = {
                "requested_model_id": mid,
                "resolved_model_id": resolved_id,
                "force": bool(force),
            }

            backend_name = self._detect_backend_name(backend) or "unknown"
            detail["backend"] = backend_name

            try:
                # External backends: do not flip model_loaded.
                if backend_name in ("llamacpp", "remote"):
                    self._ms.set_loaded_model_id(resolved_id)
                    self._ms.set_model_loaded(False)
                    self._emit_state_gauges()
                    return LoadResult(
                        model_id=resolved_id,
                        loaded=False,
                        load_mode=mode,
                        detail={**detail, "status": "noop_external"},
                    )

                # Transformers: load weights.
                is_loaded_fn = getattr(backend, "is_loaded", None)
                already_loaded = False
                if callable(is_loaded_fn):
                    try:
                        already_loaded = bool(is_loaded_fn())
                    except Exception:
                        already_loaded = False

                if force or not already_loaded:
                    await self._ensure_loaded_async(backend)

                loaded_now = True
                if callable(is_loaded_fn):
                    try:
                        loaded_now = bool(is_loaded_fn())
                    except Exception:
                        loaded_now = True

                self._ms.set_loaded_model_id(resolved_id)
                self._ms.set_model_loaded(bool(loaded_now))
                self._emit_state_gauges()

                return LoadResult(
                    model_id=resolved_id,
                    loaded=bool(loaded_now),
                    load_mode=mode,
                    detail={**detail, "status": "loaded_in_process" if loaded_now else "load_attempted"},
                )

            except AppError as e:
                self._ms.set_model_loaded(False)
                self._ms.set_loaded_model_id(None)
                self._ms.set_model_error(getattr(e, "message", None) or str(e))
                self._emit_state_gauges()
                self._metric_inc(LLM_LOADER_OPS_FAIL_TOTAL, op=op, reason=e.code if hasattr(e, "code") else "AppError")
                raise
            except Exception as e:
                msg = f"{type(e).__name__}: {e}"
                self._ms.set_model_error(msg)
                self._ms.set_model_loaded(False)
                self._ms.set_loaded_model_id(None)
                self._emit_state_gauges()
                logger.exception("runtime model load failed: model_id=%s", resolved_id)
                self._metric_inc(LLM_LOADER_OPS_FAIL_TOTAL, op=op, reason=type(e).__name__)
                raise AppError(
                    code="model_load_failed",
                    message="Failed to load model",
                    status_code=500,
                    extra={**detail, "error": msg},
                ) from e
            finally:
                self._metric_obs(LLM_LOADER_OP_LATENCY_SECONDS, time.perf_counter() - t0, op=op)

    async def load_default(self, *, force: bool = False) -> LoadResult:
        op = "load_default"
        t0 = time.perf_counter()
        self._metric_inc(LLM_LOADER_OPS_TOTAL, op=op)

        try:
            llm = await self._ensure_llm_exists()
            snap = self._ms.snapshot()

            if snap.runtime_default_model_id:
                return await self.load_model(snap.runtime_default_model_id, force=force)

            if isinstance(llm, MultiModelManager):
                return await self.load_model(llm.default_id, force=force)

            return await self.load_model(getattr(llm, "model_id", None) or "default", force=force)
        except Exception as e:
            self._metric_inc(LLM_LOADER_OPS_FAIL_TOTAL, op=op, reason=type(e).__name__)
            raise
        finally:
            self._metric_obs(LLM_LOADER_OP_LATENCY_SECONDS, time.perf_counter() - t0, op=op)

    async def load_all_enabled_models(self, *, force: bool = False) -> Dict[str, LoadResult]:
        op = "load_all_enabled_models"
        t0 = time.perf_counter()
        self._metric_inc(LLM_LOADER_OPS_TOTAL, op=op)

        try:
            async with self._lock:
                self._ms.set_model_error(None)
                llm = await self._ensure_llm_exists()

                if not isinstance(llm, MultiModelManager):
                    r = await self.load_default(force=force)
                    return {r.model_id: r}

                results: Dict[str, LoadResult] = {}
                for mid in llm.list_models():
                    try:
                        r = await self.load_model(mid, force=force)
                        results[mid] = r
                    except Exception as e:
                        msg = f"{type(e).__name__}: {e}"
                        self._ms.set_model_error(msg)
                        results[mid] = LoadResult(
                            model_id=mid,
                            loaded=False,
                            load_mode=self._get_model_load_mode(),
                            detail={"status": "failed", "error": msg},
                        )

                self._emit_state_gauges()
                return results
        except Exception as e:
            self._metric_inc(LLM_LOADER_OPS_FAIL_TOTAL, op=op, reason=type(e).__name__)
            raise
        finally:
            self._metric_obs(LLM_LOADER_OP_LATENCY_SECONDS, time.perf_counter() - t0, op=op)

    async def set_default_model(self, model_id: str) -> Dict[str, Any]:
        op = "set_default_model"
        t0 = time.perf_counter()
        self._metric_inc(LLM_LOADER_OPS_TOTAL, op=op)

        mid = (model_id or "").strip()
        if not mid:
            self._metric_inc(LLM_LOADER_OPS_FAIL_TOTAL, op=op, reason="invalid_request")
            raise AppError(code="invalid_request", message="model_id is required", status_code=400)

        try:
            async with self._lock:
                llm = await self._ensure_llm_exists()
                if isinstance(llm, MultiModelManager) and mid not in llm:
                    raise AppError(
                        code="model_missing",
                        message=f"Model '{mid}' not found in LLM registry",
                        status_code=400,
                        extra={"available": llm.list_models(), "default_id": llm.default_id},
                    )
                self._ms.set_runtime_default_model_id(mid)
                self._emit_state_gauges()
                return {"default_model": mid, "persisted": False}
        except Exception as e:
            self._metric_inc(LLM_LOADER_OPS_FAIL_TOTAL, op=op, reason=type(e).__name__)
            raise
        finally:
            self._metric_obs(LLM_LOADER_OP_LATENCY_SECONDS, time.perf_counter() - t0, op=op)

    async def clear_runtime_default(self) -> Dict[str, Any]:
        op = "clear_runtime_default"
        t0 = time.perf_counter()
        self._metric_inc(LLM_LOADER_OPS_TOTAL, op=op)

        try:
            async with self._lock:
                self._ms.set_runtime_default_model_id(None)
                self._emit_state_gauges()
                return {"default_model": None, "persisted": False}
        except Exception as e:
            self._metric_inc(LLM_LOADER_OPS_FAIL_TOTAL, op=op, reason=type(e).__name__)
            raise
        finally:
            self._metric_obs(LLM_LOADER_OP_LATENCY_SECONDS, time.perf_counter() - t0, op=op)

    async def probe_model(self, model_id: str) -> ProbeResult:
        op = "probe_model"
        t0 = time.perf_counter()
        self._metric_inc(LLM_LOADER_OPS_TOTAL, op=op)

        mid = (model_id or "").strip()
        if not mid:
            self._metric_inc(LLM_LOADER_OPS_FAIL_TOTAL, op=op, reason="invalid_request")
            raise AppError(code="invalid_request", message="model_id is required", status_code=400)

        try:
            async with self._lock:
                llm = await self._ensure_llm_exists()
                resolved_id, backend = self._resolve_backend(llm, mid)

                detail: Dict[str, Any] = {
                    "requested_model_id": mid,
                    "resolved_model_id": resolved_id,
                    "backend": self._detect_backend_name(backend),
                }

                fn_probe = getattr(backend, "probe", None)
                if callable(fn_probe):
                    try:
                        res = await anyio.to_thread.run_sync(fn_probe)
                        if isinstance(res, dict):
                            ok = bool(res.get("ok", True))
                            return ProbeResult(model_id=resolved_id, ok=ok, detail={**detail, **res})
                        return ProbeResult(model_id=resolved_id, ok=bool(res), detail={**detail, "probe": bool(res)})
                    except Exception as e:
                        msg = f"{type(e).__name__}: {e}"
                        self._metric_inc(LLM_LOADER_OPS_FAIL_TOTAL, op=op, reason=type(e).__name__)
                        return ProbeResult(model_id=resolved_id, ok=False, detail={**detail, "status": "failed", "error": msg})

                fn_loaded = getattr(backend, "is_loaded", None)
                if callable(fn_loaded):
                    try:
                        ok = bool(fn_loaded())
                        return ProbeResult(model_id=resolved_id, ok=ok, detail={**detail, "is_loaded": ok})
                    except Exception as e:
                        msg = f"{type(e).__name__}: {e}"
                        self._metric_inc(LLM_LOADER_OPS_FAIL_TOTAL, op=op, reason=type(e).__name__)
                        return ProbeResult(model_id=resolved_id, ok=False, detail={**detail, "status": "failed", "error": msg})

                return ProbeResult(model_id=resolved_id, ok=True, detail={**detail, "status": "unknown_ok"})
        finally:
            self._metric_obs(LLM_LOADER_OP_LATENCY_SECONDS, time.perf_counter() - t0, op=op)

    async def status(self) -> Dict[str, Any]:
        """
        Consolidated runtime status snapshot for admin/UI/debug.
        """
        op = "status"
        t0 = time.perf_counter()
        self._metric_inc(LLM_LOADER_OPS_TOTAL, op=op)

        try:
            llm = self._get_llm()
            cfg = self._get_models_config()

            registry_list = None
            if isinstance(llm, MultiModelManager):
                try:
                    registry_list = [s.__dict__ for s in llm.status()]
                except Exception:
                    registry_list = None

            snap = self._ms.snapshot()
            self._emit_state_gauges()

            return {
                "model_state": {
                    "model_error": snap.model_error,
                    "model_load_mode": snap.model_load_mode,
                    "model_loaded": snap.model_loaded,
                    "loaded_model_id": snap.loaded_model_id,
                    "runtime_default_model_id": snap.runtime_default_model_id,
                },
                "models_config_loaded": bool(cfg is not None),
                "registry_kind": type(llm).__name__ if llm is not None else None,
                "registry": registry_list,
            }
        finally:
            self._metric_obs(LLM_LOADER_OP_LATENCY_SECONDS, time.perf_counter() - t0, op=op)