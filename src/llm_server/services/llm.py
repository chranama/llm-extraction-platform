# src/llm_server/services/llm.py
from __future__ import annotations

import os
import threading
from typing import Iterator, Optional, List, Dict, Any

import torch
import yaml
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)

from llm_server.core.config import settings
from llm_server.services.llm_api import HttpLLMClient


# -----------------------------------
# Configuration helpers
# -----------------------------------

# Map strings to torch dtypes
DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

# Device selection
# - If settings.model_device is set, prefer that.
# - Otherwise use MPS if available, else CPU.
if settings.model_device:
    DEVICE = settings.model_device
else:
    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Default stops to avoid the model continuing a new user turn
DEFAULT_STOPS: List[str] = ["\nUser:", "\nuser:", "User:", "###"]


# ===================================
# MULTI MODEL MANAGER
# ===================================

class MultiModelManager:
    """
    Holds and routes between many LLM backends.

    Backends can be:
    - Local ModelManager
    - HttpLLMClient
    - DummyModelManager (tests)

    All must expose: generate(...) and stream(...)
    """

    def __init__(self, models: Dict[str, Any], default_id: str):
        self._models = models
        self.default_id = default_id

    @property
    def models(self) -> Dict[str, Any]:
        return self._models

    def get(self, model_id: str) -> Any:
        return self._models[model_id]

    def __getitem__(self, model_id: str) -> Any:
        return self._models[model_id]

    def __contains__(self, model_id: object) -> bool:
        return model_id in self._models

    def list_models(self) -> List[str]:
        return list(self._models.keys())

    def ensure_loaded(self) -> None:
        """
        Used by /readyz health check.

        Only ensures the *default* model is loaded so you don't blow up VRAM
        by loading everything at once.
        """
        if self.default_id in self._models:
            mgr = self._models[self.default_id]
            if hasattr(mgr, "ensure_loaded"):
                mgr.ensure_loaded()

    def load_all(self) -> None:
        """
        Eagerly load all models. Use with care (VRAM!).
        """
        for mgr in self._models.values():
            if hasattr(mgr, "ensure_loaded"):
                mgr.ensure_loaded()


# ------------------------------------------------------------
# YAML-based model config loader
# ------------------------------------------------------------

def _load_models_from_yaml() -> tuple[str, List[str]] | None:
    """
    Try to load default_model + models list from models.yaml (or custom path).

    Returns:
        (default_model_id, all_model_ids) or None on any issue.
    """
    path = getattr(settings, "models_config_path", None) or "models.yaml"

    if not path or not os.path.exists(path):
        return None

    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        # If YAML is malformed, just fall back to env-based config
        return None

    default_model = data.get("default_model")
    models_list = data.get("models") or []

    # Extract ids
    ids: List[str] = []
    for m in models_list:
        if isinstance(m, dict) and "id" in m:
            ids.append(str(m["id"]))
        elif isinstance(m, str):
            ids.append(m)

    if not ids and not default_model:
        return None

    # If default_model not provided, use first id
    if not default_model:
        default_model = ids[0]

    if default_model not in ids:
        ids.insert(0, default_model)

    return default_model, ids


def build_llm_from_settings() -> Any:
    """
    Creates the appropriate LLM backend(s) from config.

    Returns either:
    - ModelManager (single model)
    - OR MultiModelManager (local + remote mix)

    Priority:
    1) If models.yaml (or models_config_path) exists and is valid:
       - Use it to define default + allowed models.
    2) Else fall back to SETTINGS:
       - settings.model_id + settings.allowed_models
    """

    # 1) Try to load from models.yaml
    yaml_cfg = _load_models_from_yaml()
    if yaml_cfg is not None:
        primary_id, all_ids = yaml_cfg

        # Sync settings so generate.py resolve_model sees the same IDs
        settings.model_id = primary_id
        settings.allowed_models = all_ids
    else:
        # 2) Fall back to environment-based config
        primary_id = settings.model_id
        all_ids = settings.all_model_ids

    # --- Only one model → just use local ModelManager ---
    if len(all_ids) == 1:
        # local single model
        return ModelManager.from_settings(settings)

    # --- Multiple models → route ---
    models: Dict[str, Any] = {}

    # Local primary model
    local = ModelManager.from_settings(settings)
    # Ensure the local manager uses the selected primary id
    local.model_id = primary_id
    models[primary_id] = local

    # Remote models via HTTP for the others
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
    """
    Local HuggingFace model backend

    - Lazy loading
    - Non-streaming generation
    - Streaming generation
    """

    model_id: str = settings.model_id
    _tokenizer = None
    _model = None
    _device = DEVICE
    _dtype = DTYPE_MAP.get(settings.model_dtype, torch.float16)

    # ---------- factory ----------
    @classmethod
    def from_settings(cls, cfg) -> "ModelManager":
        """
        Build a ModelManager from Settings.

        Uses:
        - cfg.model_id
        - cfg.model_dtype
        - cfg.model_device (if set) or the global DEVICE
        """
        instance = cls()
        instance.model_id = cfg.model_id
        instance._dtype = DTYPE_MAP.get(
            getattr(cfg, "model_dtype", "float16"),
            torch.float16,
        )
        if getattr(cfg, "model_device", None):
            instance._device = cfg.model_device
        else:
            instance._device = DEVICE
        return instance

    # ---------- lifecycle ----------
    def ensure_loaded(self) -> None:
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

        if self._model is None:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=self._dtype,
                device_map=None,
            )
            self._model.to(self._device)
            self._model.eval()

    # ---------- helpers ----------
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

    # ---------- non-streaming ----------
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
        """Return the full completion as a single string."""
        self.ensure_loaded()

        tok = self._tokenizer
        model = self._model

        # Apply default stops if none provided
        stops = stop if (stop and len(stop) > 0) else DEFAULT_STOPS

        inputs = tok(prompt, return_tensors="pt").to(self._device)

        # Normalize sampling params so they’re safe
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
        tail = text[len(prompt):]

        return self._truncate_on_stop(tail, stops)

    # ---------- streaming ----------
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
        """Yield text chunks as they are generated using TextIteratorStreamer."""
        self.ensure_loaded()

        tok = self._tokenizer
        model = self._model

        # Apply default stops if none provided
        stops = stop if (stop and len(stop) > 0) else DEFAULT_STOPS

        inputs = tok(prompt, return_tensors="pt").to(self._device)

        streamer = TextIteratorStreamer(
            tok,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        # Normalize sampling params so they’re safe
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
        for piece in streamer:
            buffer += piece
            # stop if any stop sequence appears
            cut_positions = [buffer.find(s) for s in stops if s in buffer]
            if cut_positions:
                cut = min(cut_positions)
                if cut >= 0:
                    if cut > 0:
                        yield buffer[:cut]
                    t.join()
                    return
            if piece:
                yield piece

        t.join()