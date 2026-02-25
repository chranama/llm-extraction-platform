from __future__ import annotations

import os
from typing import Optional

import httpx
import pytest

from llm_eval.client.http_client import HttpEvalClient

DEFAULT_INTEGRATION_BASE_URL = "http://localhost:8000"
DEFAULT_INTEGRATION_API_KEY = "test_api_key_123"


def _get_env(name: str) -> Optional[str]:
    v = os.getenv(name)
    if v is None:
        return None
    v = v.strip()
    return v or None


@pytest.fixture(scope="session")
def integration_base_url() -> str:
    """
    Base URL for llm-server instance used by integration tests.

    Priority:
      - INTEGRATION_BASE_URL
      - LLM_SERVER_BASE_URL
      - default localhost
    """
    v = (
        _get_env("INTEGRATION_BASE_URL")
        or _get_env("LLM_SERVER_BASE_URL")
        or DEFAULT_INTEGRATION_BASE_URL
    )
    return v.rstrip("/")


@pytest.fixture(scope="session")
def integration_api_key() -> str:
    """
    API key for llm-server used by integration tests.

    Priority:
      - API_KEY
      - default local test key
    """
    return _get_env("API_KEY") or DEFAULT_INTEGRATION_API_KEY


@pytest.fixture(scope="session")
def require_live_server(integration_base_url: str, integration_api_key: str) -> None:
    """
    Skip tests that require a reachable live server.
    """
    try:
        r = httpx.get(
            f"{integration_base_url}/modelz",
            headers={"X-API-Key": integration_api_key},
            timeout=3.0,
        )
        # Any HTTP response implies server is reachable.
        _ = r.status_code
    except Exception as e:
        pytest.skip(f"Live server not reachable at {integration_base_url}: {type(e).__name__}: {e}")


@pytest.fixture
def live_client(integration_base_url: str, integration_api_key: str) -> HttpEvalClient:
    return HttpEvalClient(base_url=integration_base_url, api_key=integration_api_key, timeout=60.0)
