# src/llm_server/services/llm.py
from __future__ import annotations

import os
import pwd
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import transformers as tf

from llm_server.core.config import get_settings
from llm_server.core.errors import AppError
from llm_server.services.llm_api import HttpLLMClient
from llm_server.services.llm_config import load_models_config
from llm_server.services.llm_registry import MultiModelManager

# -----------------------------------
# Configuration helpers (local)
# -----------------------------------

DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

DEFAULT_STOPS: List[str] = ["\nUser:", "\nuser:", "User:", "###"]


def _real_user_home() -> str:
    try:
        return pwd.getpwuid(os.getuid()).pw_dir
    except Exception:
        return os.path.expanduser("~")


def _device_from_settings(cfg) -> str:
    # explicit override wins
    dev = getattr(cfg, "model_device", None)
    if isinstance(dev, str) and dev.strip():
        return dev.strip()
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _resolve_hf_home(cfg) -> str:
    cfg_val = getattr(cfg, "hf_home", None)
    if isinstance(cfg_val, str) and cfg_val.strip():
        return cfg_val.strip()

    env_val = os.environ.get("HF_HOME")
    if env_val and env_val.strip():
        return env_val.strip()

    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg and xdg.strip():
        return os.path.join(xdg, "huggingface")

    return os.path.join(_real_user_home(), ".cache", "huggingface")


def _configure_hf_cache_env(cfg) -> dict[str, str]:
    hf_home = _resolve_hf_home(cfg)
    hub_cache = os.environ.get("HF_HUB_CACHE") or os.path.join(hf_home, "hub")

    os.environ["HF_HOME"] = hf_home
    os.environ["HF_HUB_CACHE"] = hub_cache
    os.environ["TRANSFORMERS_CACHE"] = os.environ.get("TRANSFORMERS_CACHE") or hub_cache
    os.environ["XDG_CACHE_HOME"] = os.environ.get("XDG_CACHE_HOME") or os.path.dirname(hf_home)

    try:
        os.makedirs(hf_home, exist_ok=True)
        os.makedirs(hub_cache, exist_ok=True)
    except Exception as e:
        raise AppError(
            code="hf_cache_unwritable",
            message="Hugging Face cache directory is not writable",
            status_code=500,
            extra={
                "hf_home": hf_home,
                "hf_hub_cache": hub_cache,
                "error": str(e),
                "env_HOME": os.environ.get("HOME"),
                "real_user_home": _real_user_home(),
            },
        ) from e

    return {
        "hf_home": hf_home,
        "hf_hub_cache": hub_cache,
        "transformers_cache": os.environ.get("TRANSFORMERS_CACHE", ""),
        "env_HOME": os.environ.get("HOME", ""),
        "real_user_home": _real_user_home(),
    }


# ===================================
# LOCAL MODEL MANAGER
# ===================================


class ModelManager:
    """
    Local (in-process) HF Transformers backend.

    Lazy-loads: ensure_loaded() is only called by:
      - app startup in eager mode
      - right before generate()
    """

    def __init__(
        self,
        *,
        model_id: str,
        device: str,
        dtype: torch.dtype,
    ) -> None:
        self.model_id: str = model_id
        self._device: str = device
        self._dtype: torch.dtype = dtype
        self._tokenizer = None
        self._model = None

    @classmethod
    def from_settings(cls, cfg) -> "ModelManager":
        dtype_str = getattr(cfg, "model_dtype", "float16")
        dtype = DTYPE_MAP.get(dtype_str, torch.float16)
        device = _device_from_settings(cfg)
        model_id = getattr(cfg, "model_id", "mistralai/Mistral-7B-v0.1")
        return cls(model_id=model_id, device=device, dtype=dtype)

    def _err_ctx(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "device": str(self._device),
            "dtype": str(self._dtype),
            "env_HOME": os.environ.get("HOME"),
            "HF_HOME": os.environ.get("HF_HOME"),
            "HF_HUB_CACHE": os.environ.get("HF_HUB_CACHE"),
            "TRANSFORMERS_CACHE": os.environ.get("TRANSFORMERS_CACHE"),
            "XDG_CACHE_HOME": os.environ.get("XDG_CACHE_HOME"),
            "real_user_home": _real_user_home(),
        }

    def is_loaded(self) -> bool:
        return (self._tokenizer is not None) and (self._model is not None)

    def ensure_loaded(self) -> None:
        try:
            cfg = get_settings()
            cache_ctx = _configure_hf_cache_env(cfg)
            cache_dir = cache_ctx["hf_hub_cache"]

            if self._tokenizer is None:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_id,
                    use_fast=True,
                    cache_dir=cache_dir,
                )
                if getattr(self._tokenizer, "pad_token", None) is None:
                    self._tokenizer.pad_token = self._tokenizer.eos_token

            if self._model is None:
                hf_cfg = AutoConfig.from_pretrained(self.model_id)

                dtype = self._dtype
                # MPS + bf16 is generally problematic
                if str(self._device) == "mps" and dtype == torch.bfloat16:
                    dtype = torch.float16

                try:
                    self._model = AutoModelForCausalLM.from_pretrained(
                        self.model_id,
                        torch_dtype=dtype,
                        low_cpu_mem_usage=True,
                        cache_dir=cache_dir,
                    )
                except ValueError:
                    archs = getattr(hf_cfg, "architectures", None) or []
                    if not archs:
                        raise
                    arch = archs[0]
                    model_cls = getattr(tf, arch, None)
                    if model_cls is None:
                        raise
                    self._model = model_cls.from_pretrained(
                        self.model_id,
                        torch_dtype=dtype,
                        low_cpu_mem_usage=True,
                        cache_dir=cache_dir,
                    )

                self._model.to(self._device)
                self._model.eval()

        except AppError:
            raise
        except Exception as e:
            raise AppError(
                code="model_load_failed",
                message="Failed to load local model",
                status_code=500,
                extra={**self._err_ctx(), "error": str(e)},
            ) from e

    @staticmethod
    def _truncate_on_stop(text: str, stop: Optional[List[str]]) -> str:
        if not stop:
            return text
        cut_positions = [text.find(s) for s in stop if s in text]
        if cut_positions:
            cut = min(cut_positions)
            if cut >= 0:
                return text[:cut]
        return text

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int | None = 0,
        repetition_penalty: float = 1.0,
        stop: Optional[List[str]] = None,
    ) -> str:
        try:
            self.ensure_loaded()

            tok = self._tokenizer
            model = self._model
            if tok is None or model is None:
                raise RuntimeError("Model not loaded")

            stops = stop if (stop and len(stop) > 0) else DEFAULT_STOPS
            inputs = tok(prompt, return_tensors="pt").to(self._device)

            use_top_k = top_k if (top_k is not None and top_k > 0) else None
            use_temperature = temperature if (temperature is not None and temperature > 0) else 0.0
            use_top_p = top_p if top_p is not None else 0.95

            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=use_temperature > 0,
                temperature=use_temperature,
                top_p=use_top_p,
                top_k=use_top_k,
                repetition_penalty=repetition_penalty,
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.pad_token_id,
            )[0]

            text = tok.decode(output_ids, skip_special_tokens=True)
            tail = text[len(prompt) :]
            return self._truncate_on_stop(tail, stops)

        except AppError:
            raise
        except Exception as e:
            raise AppError(
                code="model_generate_failed",
                message="Local model generation failed",
                status_code=500,
                extra={**self._err_ctx(), "error": str(e)},
            ) from e


# ===================================
# BUILDER / WIRING
# ===================================


def build_llm_from_settings() -> Any:
    """
    Construct the service LLM backend.

    Behavior:
      - If exactly 1 model id -> local ModelManager
      - If multiple model ids:
          - primary is local ModelManager
          - non-primary are remote HttpLLMClient (requires llm_service_url)
          - wrapped in MultiModelManager registry

    NOTE: MultiModelManager.ensure_loaded() loads only the default model
    (cloud-friendly). Non-default models remain lazy until requested.
    """
    cfg = load_models_config()
    primary_id = cfg.primary_id
    all_ids = cfg.model_ids

    s = get_settings()

    if len(all_ids) == 1:
        mgr = ModelManager.from_settings(s)
        mgr.model_id = primary_id
        return mgr

    if not getattr(s, "llm_service_url", None):
        raise AppError(
            code="remote_models_require_llm_service_url",
            message="Multiple models configured but llm_service_url is not set",
            status_code=500,
            extra={"primary_id": primary_id, "all_model_ids": all_ids},
        )

    models: Dict[str, Any] = {}
    meta: Dict[str, Dict[str, Any]] = {}

    local = ModelManager.from_settings(s)
    local.model_id = primary_id
    models[primary_id] = local
    meta[primary_id] = {"backend": "local_hf", "load_mode": "lazy(default eager-by-lifespan)"}

    for mid in all_ids:
        if mid == primary_id:
            continue
        models[mid] = HttpLLMClient(
            base_url=s.llm_service_url,
            model_id=mid,
            timeout=s.http_client_timeout,
        )
        meta[mid] = {"backend": "http_remote", "load_mode": "remote"}

    return MultiModelManager(models=models, default_id=primary_id, model_meta=meta)