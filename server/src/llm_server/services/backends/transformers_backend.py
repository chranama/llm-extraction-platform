# server/src/llm_server/services/backends/transformers_backend.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

from llm_server.core.errors import AppError
from llm_server.services.backends.base import GenerateResult, GenerateTimings, GenerateUsage, LLMBackend


@dataclass(frozen=True)
class TransformersBackendConfig:
    hf_id: str
    device: str = "auto"         # "auto"|"cpu"|"cuda"|"mps"
    dtype: Optional[str] = None  # "float16"|"bfloat16"|"float32"|None
    trust_remote_code: bool = False

    # Optional generation defaults
    default_temperature: float = 0.7
    default_top_p: float = 0.95


class TransformersBackend(LLMBackend):
    backend_name: str = "transformers"

    def __init__(self, *, model_id: str, cfg: TransformersBackendConfig) -> None:
        self.model_id = model_id
        self._cfg = cfg

        self._loaded = False
        self._tok = None
        self._model = None

    def ensure_loaded(self) -> None:
        if self._loaded:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as e:
            raise AppError(
                code="backend_missing_deps",
                message="Transformers backend dependencies missing",
                status_code=500,
                extra={"error": repr(e)},
            ) from e

        hf_id = (self._cfg.hf_id or "").strip()
        if not hf_id:
            raise AppError(
                code="backend_config_invalid",
                message="Transformers backend requires hf_id",
                status_code=500,
                extra={"model_id": self.model_id},
            )

        # dtype mapping (optional)
        torch_dtype = None
        if isinstance(self._cfg.dtype, str):
            d = self._cfg.dtype.strip().lower()
            if d == "float16":
                torch_dtype = torch.float16
            elif d == "bfloat16":
                torch_dtype = torch.bfloat16
            elif d == "float32":
                torch_dtype = torch.float32

        tok = AutoTokenizer.from_pretrained(
            hf_id,
            use_fast=True,
            trust_remote_code=bool(self._cfg.trust_remote_code),
        )
        model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            torch_dtype=torch_dtype,
            trust_remote_code=bool(self._cfg.trust_remote_code),
        )
        model.eval()

        # device placement
        dev = (self._cfg.device or "auto").strip().lower()
        if dev == "auto":
            if torch.cuda.is_available():
                dev = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():  # type: ignore[attr-defined]
                dev = "mps"
            else:
                dev = "cpu"

        try:
            model.to(dev)
        except Exception:
            # If move fails, fall back to cpu (never crash startup from here)
            model.to("cpu")
            dev = "cpu"

        self._tok = tok
        self._model = model
        self._loaded = True

    def generate(
        self,
        *,
        prompt: str,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> str:
        r = self.generate_rich(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=stop,
            **kwargs,
        )
        return r.text

    def generate_rich(
        self,
        *,
        prompt: str,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> GenerateResult:
        if not isinstance(prompt, str) or not prompt.strip():
            raise AppError(code="invalid_request", message="prompt must be a non-empty string", status_code=400)

        self.ensure_loaded()
        tok = self._tok
        model = self._model
        if tok is None or model is None:
            raise AppError(code="backend_not_ready", message="Transformers backend not loaded", status_code=503)

        try:
            import torch
        except Exception as e:
            raise AppError(code="backend_missing_deps", message="torch missing", status_code=500, extra={"error": repr(e)}) from e

        # defaults
        temp = float(temperature) if isinstance(temperature, (int, float)) else float(self._cfg.default_temperature)
        tp = float(top_p) if isinstance(top_p, (int, float)) else float(self._cfg.default_top_p)
        mnt = int(max_new_tokens) if isinstance(max_new_tokens, int) and max_new_tokens > 0 else 256

        # Tokenize and run generation
        t0 = time.perf_counter()
        inputs = tok(prompt, return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": mnt,
            "do_sample": bool(temp > 0),
            "temperature": temp,
            "top_p": tp,
        }
        if isinstance(top_k, int) and top_k > 0:
            gen_kwargs["top_k"] = int(top_k)

        # NOTE: stop handling is non-trivial in pure HF generate(). Keeping signature stable; improve later if needed.
        _ = stop

        with torch.no_grad():
            out_ids = model.generate(**inputs, **gen_kwargs)

        dt_ms = (time.perf_counter() - t0) * 1000.0

        # Decode only the new tokens
        in_len = int(inputs["input_ids"].shape[-1])
        gen_ids = out_ids[0][in_len:]
        text = tok.decode(gen_ids, skip_special_tokens=True)

        return GenerateResult(
            text=text,
            usage=GenerateUsage(),  # route does token counting already
            timings=GenerateTimings(total_ms=dt_ms, backend_ms=dt_ms),
            raw=None,
        )