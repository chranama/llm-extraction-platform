from __future__ import annotations

import httpx
from typing import Optional


class HttpEvalClient:
    """
    Talks to the llm-server /v1/generate endpoint.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: str = "test_api_key_123",
        timeout: float = 60.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    async def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:
        payload = {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        }

        headers = {"X-API-Key": self.api_key}

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(
                f"{self.base_url}/v1/generate",
                json=payload,
                headers=headers,
            )

            r.raise_for_status()
            data = r.json()

        return data["output"]