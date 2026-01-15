import os
import pytest
import httpx

DEFAULT_BASE_URL = "http://localhost:8080/api"  # nginx route in your stack

@pytest.fixture(scope="session")
def base_url() -> str:
    return os.environ.get("INTEGRATION_BASE_URL", DEFAULT_BASE_URL).rstrip("/")

@pytest.fixture(scope="session")
def api_key() -> str:
    return os.environ.get("API_KEY", "")

@pytest.fixture(scope="session")
def client(base_url: str, api_key: str) -> httpx.Client:
    headers = {}
    if api_key:
        headers["X-API-Key"] = api_key
    return httpx.Client(base_url=base_url, headers=headers, timeout=30.0)