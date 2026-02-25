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

    def is_loaded(self) -> bool:
        return bool(self._loaded and self._tok is not None and self._model is not None)

    def model_info(self) -> dict[str, Any]:
        """
        Best-effort local metadata snapshot.
        Never throws. Must be safe to call from readiness/health paths.
        """
        out: dict[str, Any] = {
            "ok": True,
            "backend": self.backend_name,
            "model_id": self.model_id,
            "hf_id": (self._cfg.hf_id or "").strip() or None,
            "loaded": bool(self.is_loaded()),
            "config": {
                "device": (self._cfg.device or "").strip() or None,
                "dtype": (self._cfg.dtype or "").strip() or None,
                "trust_remote_code": bool(self._cfg.trust_remote_code),
            },
        }

        # These are best-effort and should not trigger any loading.
        try:
            model = self._model
            if model is not None:
                # device + dtype reflect runtime reality
                try:
                    dev = getattr(model, "device", None)
                    out["device"] = str(dev) if dev is not None else None
                except Exception:
                    out["device"] = None

                try:
                    dt = getattr(model, "dtype", None)
                    out["dtype"] = str(dt) if dt is not None else None
                except Exception:
                    out["dtype"] = None

                # Some models carry config metadata
                try:
                    cfg = getattr(model, "config", None)
                    if cfg is not None:
                        name_or_path = getattr(cfg, "_name_or_path", None)
                        if isinstance(name_or_path, str) and name_or_path.strip():
                            out["hf_name_or_path"] = name_or_path.strip()
                except Exception:
                    pass

            # Library versions are helpful for correlation
            try:
                import torch  # type: ignore

                out["torch_version"] = getattr(torch, "__version__", None)
            except Exception:
                out["torch_version"] = None

            try:
                import transformers  # type: ignore

                out["transformers_version"] = getattr(transformers, "__version__", None)
            except Exception:
                out["transformers_version"] = None

        except Exception as e:
            # Never throw; convert to degraded snapshot
            out["ok"] = False
            out["error"] = f"{type(e).__name__}: {e}"

        return out

    def ensure_loaded(self) -> None:
        if self.is_loaded():
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

        # NOTE:
        # Lazy loading is now controlled by deps.py + RuntimeModelLoader.
        # If we get here and weights are not loaded, treat as not ready.
        if not self.is_loaded():
            raise AppError(code="backend_not_ready", message="Transformers backend not loaded", status_code=503)

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

        _ = stop  # stop handling is non-trivial with HF generate()

        with torch.no_grad():
            out_ids = model.generate(**inputs, **gen_kwargs)

        dt_ms = (time.perf_counter() - t0) * 1000.0

        # Decode only the new tokens
        in_len = int(inputs["input_ids"].shape[-1])
        gen_ids = out_ids[0][in_len:]
        text = tok.decode(gen_ids, skip_special_tokens=True)

        return GenerateResult(
            text=text,
            usage=GenerateUsage(),
            timings=GenerateTimings(total_ms=dt_ms, backend_ms=dt_ms),
            raw=None,
        )