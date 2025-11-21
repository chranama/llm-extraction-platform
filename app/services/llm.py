# app/services/llm.py
from __future__ import annotations

import threading
from typing import Iterator, Optional, List

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)

from app.core.config import settings  # <-- central config import

# Map strings to torch dtypes
DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

# Device selection (Apple Silicon MPS or CPU as fallback)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Default stops to avoid the model continuing a new user turn
DEFAULT_STOPS: List[str] = ["\nUser:", "\nuser:", "User:", "###"]


class ModelManager:
    """
    Singleton-ish manager for tokenizer/model with:
    - Lazy loading
    - Non-streaming generation (return full text)
    - Streaming generation (yield chunks)
    """

    model_id: str = settings.model_id
    _tokenizer = None
    _model = None
    _device = DEVICE
    _dtype = DTYPE_MAP.get(settings.model_dtype, torch.float16)

    # ------------- lifecycle -------------
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

    # ------------- helpers -------------
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

    # ------------- non-streaming -------------
    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 0,
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

        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k if top_k > 0 else None,
            repetition_penalty=repetition_penalty,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
        )[0]

        text = tok.decode(output_ids, skip_special_tokens=True)
        tail = text[len(prompt):]
        return self._truncate_on_stop(tail, stops)

    # ------------- streaming -------------
    def stream(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 0,
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

        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k if top_k > 0 else None,
            repetition_penalty=repetition_penalty,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
            streamer=streamer,
        )

        import threading
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