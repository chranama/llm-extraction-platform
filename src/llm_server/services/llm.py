# src/llm_server/services/llm.py
from __future__ import annotations

import os
import pwd
import threading
from typing import Any, Dict, Iterator, List, Optional

import torch
import yaml
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import transformers as tf

from llm_server.core.config import settings
from llm_server.core.errors import AppError
from llm_server.services.llm_api import HttpLLMClient

# -----------------------------------
# Configuration helpers
# -----------------------------------

DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

if settings.model_device:
    DEVICE = settings.model_device
else:
    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

DEFAULT_STOPS: List[str] = ["\nUser:", "\nuser:", "User:", "###"]


def _real_user_home() -> str:
    """
    Prefer the *actual* user's home directory even if HOME is wrong.
    This avoids falling back to /root in odd env situations.
    """
    try:
        return pwd.getpwuid(os.getuid()).pw_dir
    except Exception:
        # last resort
        return os.path.expanduser("~")


def _resolve_hf_home() -> str:
    """
    Resolve HF cache root deterministically.

    Priority:
      1) settings.hf_home (if you add it to config)
      2) HF_HOME env
      3) XDG_CACHE_HOME/huggingface
      4) <real_user_home>/.cache/huggingface
    """
    # Optional: if you later add hf_home to your Settings
    cfg_val = getattr(settings, "hf_home", None)
    if isinstance(cfg_val, str) and cfg_val.strip():
        return cfg_val

    env_val = os.environ.get("HF_HOME")
    if env_val and env_val.strip():
        return env_val

    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg and xdg.strip():
        return os.path.join(xdg, "huggingface")

    return os.path.join(_real_user_home(), ".cache", "huggingface")


def _configure_hf_cache_env() -> dict[str, str]:
    """
    Set HF/Transformers cache env vars *early* and ensure directories exist.
    Returns useful debug context.
    """
    hf_home = _resolve_hf_home()
    hub_cache = os.environ.get("HF_HUB_CACHE") or os.path.join(hf_home, "hub")

    # Force these so HF libs don't invent /root paths
    os.environ["HF_HOME"] = hf_home
    os.environ["HF_HUB_CACHE"] = hub_cache
    os.environ["TRANSFORMERS_CACHE"] = os.environ.get("TRANSFORMERS_CACHE") or hub_cache

    # Some libs use XDG_CACHE_HOME; set it to the parent of HF_HOME
    os.environ["XDG_CACHE_HOME"] = os.environ.get("XDG_CACHE_HOME") or os.path.dirname(hf_home)

    # Ensure dirs exist
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
# MULTI MODEL MANAGER
# ===================================


class MultiModelManager:
    def __init__(self, models: Dict[str, Any], default_id: str):
        self._models = models
        self.default_id = default_id

    @property
    def models(self) -> Dict[str, Any]:
        return self._models

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
        return model_id in self._models

    def list_models(self) -> List[str]:
        return list(self._models.keys())

    def ensure_loaded(self) -> None:
        mgr = self._models.get(self.default_id)
        if mgr is None:
            raise AppError(
                code="default_model_missing",
                message="Default model is missing from LLM registry",
                status_code=500,
                extra={"default_id": self.default_id, "available": self.list_models()},
            )
        if hasattr(mgr, "ensure_loaded"):
            mgr.ensure_loaded()

    def load_all(self) -> None:
        for mgr in self._models.values():
            if hasattr(mgr, "ensure_loaded"):
                mgr.ensure_loaded()

    def is_loaded(self) -> bool:
        mgr = self._models.get(self.default_id)
        if mgr is None:
            return False

        fn = getattr(mgr, "is_loaded", None)
        if callable(fn):
            try:
                return bool(fn())
            except Exception:
                return False

        m = getattr(mgr, "_model", None)
        t = getattr(mgr, "_tokenizer", None)
        return (m is not None) and (t is not None)


# ------------------------------------------------------------
# YAML-based model config loader
# ------------------------------------------------------------


def _models_yaml_path() -> str:
    return str(getattr(settings, "models_config_path", None) or "models.yaml")


def _load_models_from_yaml() -> tuple[str, List[str]] | None:
    path = _models_yaml_path()
    if not path or not os.path.exists(path):
        return None

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

    default_model = data.get("default_model")
    models_list = data.get("models") or []

    if default_model is not None and not isinstance(default_model, str):
        raise AppError(
            code="models_yaml_invalid",
            message="models.yaml default_model must be a string",
            status_code=500,
            extra={"path": path},
        )

    if models_list is not None and not isinstance(models_list, list):
        raise AppError(
            code="models_yaml_invalid",
            message="models.yaml models must be a list",
            status_code=500,
            extra={"path": path},
        )

    ids: List[str] = []
    for m in models_list:
        if isinstance(m, dict) and "id" in m:
            mid = m["id"]
            if not isinstance(mid, str):
                raise AppError(
                    code="models_yaml_invalid",
                    message="models.yaml model id must be a string",
                    status_code=500,
                    extra={"path": path, "bad_item": str(m)},
                )
            ids.append(mid)
        elif isinstance(m, str):
            ids.append(m)
        else:
            raise AppError(
                code="models_yaml_invalid",
                message="models.yaml models entries must be strings or objects with an 'id' field",
                status_code=500,
                extra={"path": path, "bad_item": str(m)},
            )

    if not ids and not default_model:
        raise AppError(
            code="models_yaml_invalid",
            message="models.yaml must define default_model and/or at least one model id in models",
            status_code=500,
            extra={"path": path},
        )

    if not default_model:
        default_model = ids[0]

    if default_model not in ids:
        ids.insert(0, default_model)

    return str(default_model), [str(x) for x in ids]


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for x in items:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def build_llm_from_settings() -> Any:
    yaml_cfg = _load_models_from_yaml()
    if yaml_cfg is not None:
        primary_id, all_ids = yaml_cfg
    else:
        primary_id = settings.model_id
        all_ids = settings.all_model_ids

    if not primary_id or not isinstance(primary_id, str):
        raise AppError(
            code="model_config_invalid",
            message="Primary model id is missing or invalid",
            status_code=500,
            extra={"primary_id": str(primary_id)},
        )

    all_ids = [str(x) for x in (all_ids or []) if str(x).strip()]
    all_ids = _dedupe_preserve_order(all_ids)

    if not all_ids:
        raise AppError(
            code="model_config_invalid",
            message="No model ids configured",
            status_code=500,
            extra={"primary_id": primary_id},
        )

    if primary_id not in all_ids:
        all_ids.insert(0, primary_id)

    try:
        settings.model_id = primary_id  # type: ignore[attr-defined]
        settings.allowed_models = all_ids  # type: ignore[attr-defined]
    except Exception:
        pass

    if len(all_ids) == 1:
        mgr = ModelManager.from_settings(settings)
        mgr.model_id = primary_id
        return mgr

    if not getattr(settings, "llm_service_url", None):
        raise AppError(
            code="remote_models_require_llm_service_url",
            message="Multiple models configured but llm_service_url is not set",
            status_code=500,
            extra={"primary_id": primary_id, "all_model_ids": all_ids},
        )

    models: Dict[str, Any] = {}
    local = ModelManager.from_settings(settings)
    local.model_id = primary_id
    models[primary_id] = local

    for mid in all_ids:
        if mid == primary_id:
            continue
        models[mid] = HttpLLMClient(
            base_url=settings.llm_service_url,
            model_id=mid,
            timeout=settings.http_client_timeout,
        )

    return MultiModelManager(models, default_id=primary_id)


# ===================================
# LOCAL MODEL MANAGER
# ===================================


class ModelManager:
    model_id: str = settings.model_id
    _tokenizer = None
    _model = None
    _device = DEVICE
    _dtype = DTYPE_MAP.get(settings.model_dtype, torch.float16)

    @classmethod
    def from_settings(cls, cfg) -> "ModelManager":
        instance = cls()
        instance.model_id = cfg.model_id
        instance._dtype = DTYPE_MAP.get(getattr(cfg, "model_dtype", "float16"), torch.float16)
        instance._device = cfg.model_device if getattr(cfg, "model_device", None) else DEVICE
        return instance

    def _err_ctx(self) -> dict[str, Any]:
        ctx = {
            "model_id": self.model_id,
            "device": str(self._device),
            "dtype": str(self._dtype),
            "env_HOME": os.environ.get("HOME"),
        }
        # include resolved HF info if we have it
        ctx.update(
            {
                "HF_HOME": os.environ.get("HF_HOME"),
                "HF_HUB_CACHE": os.environ.get("HF_HUB_CACHE"),
                "TRANSFORMERS_CACHE": os.environ.get("TRANSFORMERS_CACHE"),
                "XDG_CACHE_HOME": os.environ.get("XDG_CACHE_HOME"),
                "real_user_home": _real_user_home(),
            }
        )
        return ctx

    def is_loaded(self) -> bool:
        return (self._tokenizer is not None) and (self._model is not None)

    def ensure_loaded(self) -> None:
        try:
            cache_ctx = _configure_hf_cache_env()
            cache_dir = cache_ctx["hf_hub_cache"]

            if self._tokenizer is None:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_id,
                    use_fast=True,
                    cache_dir=cache_dir,
                )
                if self._tokenizer.pad_token is None:
                    self._tokenizer.pad_token = self._tokenizer.eos_token

            if self._model is None:
                cfg = AutoConfig.from_pretrained(self.model_id)

                # Prefer configured dtype, but MPS + bfloat16 can be flaky depending on torch/macOS.
                dtype = self._dtype
                if str(self._device) == "mps" and dtype == torch.bfloat16:
                    # Keep your config intent, but fall back if needed.
                    dtype = torch.float16

                # Try standard causal LM first
                try:
                    self._model = AutoModelForCausalLM.from_pretrained(
                        self.model_id,
                        torch_dtype=dtype,
                        low_cpu_mem_usage=True,
                    )
                except ValueError:
                    # Fall back to explicit architecture class (e.g., Mistral3ForConditionalGeneration)
                    archs = getattr(cfg, "architectures", None) or []
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

    def stream(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int | None = 0,
        repetition_penalty: float = 1.0,
        stop: Optional[List[str]] = None,
    ) -> Iterator[str]:
        try:
            self.ensure_loaded()

            tok = self._tokenizer
            model = self._model

            stops = stop if (stop and len(stop) > 0) else DEFAULT_STOPS
            inputs = tok(prompt, return_tensors="pt").to(self._device)

            streamer = TextIteratorStreamer(
                tok,
                skip_prompt=True,
                skip_special_tokens=True,
            )

            use_top_k = top_k if (top_k is not None and top_k > 0) else None
            use_temperature = temperature if (temperature is not None and temperature > 0) else 0.0
            use_top_p = top_p if top_p is not None else 0.95

            gen_kwargs = dict(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=use_temperature > 0,
                temperature=use_temperature,
                top_p=use_top_p,
                top_k=use_top_k,
                repetition_penalty=repetition_penalty,
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.pad_token_id,
                streamer=streamer,
            )

            t = threading.Thread(target=model.generate, kwargs=gen_kwargs)
            t.start()

            buffer = ""
            emitted_len = 0

            for piece in streamer:
                if not piece:
                    continue

                buffer += piece
                cut_positions = [buffer.find(s) for s in stops if s and s in buffer]
                if cut_positions:
                    cut = min(cut_positions)
                    if cut >= 0:
                        if cut > emitted_len:
                            yield buffer[emitted_len:cut]
                        t.join()
                        return

                if len(buffer) > emitted_len:
                    yield buffer[emitted_len:]
                    emitted_len = len(buffer)

            t.join()

        except AppError:
            raise
        except Exception as e:
            raise AppError(
                code="model_stream_failed",
                message="Local model streaming failed",
                status_code=500,
                extra={**self._err_ctx(), "error": str(e)},
            ) from e