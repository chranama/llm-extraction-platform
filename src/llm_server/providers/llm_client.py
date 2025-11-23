# src/llm_server/providers/llm_client.py

from __future__ import annotations
from typing import Optional

# IMPORTANT:
# This file is a thin abstraction layer between the API and the actual LLM implementation.
# In production it imports the real client. In tests, it is monkeypatched.

try:
    # ✅ Real implementation
    from llm_server.services.llm_api import LLMApiClient as HttpLLMClient
except Exception:
    # ✅ Safe stub for local / test environments
    class HttpLLMClient:
        def __init__(self, *args, **kwargs):
            pass

        async def ensure_loaded(self):
            return None

        async def generate(
            self,
            prompt: str,
            max_new_tokens: Optional[int] = None,
            temperature: Optional[float] = None,
            top_p: Optional[float] = None,
            top_k: Optional[int] = None,
            stop: Optional[list[str]] = None,
        ) -> str:
            raise NotImplementedError(
                "HttpLLMClient is a stub. "
                "In production it must be provided by services.llm_api.LLMApiClient, "
                "and in tests it is replaced with DummyModelManager via monkeypatch."
            )