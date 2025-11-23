# app/core/limits.py
import asyncio
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

# Simple global semaphore for generation endpoints
MAX_CONCURRENT_REQUESTS = 2
_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)


class _ConcurrencyMiddleware(BaseHTTPMiddleware):
    """
    Middleware that limits concurrency for heavy endpoints.
    Here we apply it to POST /v1/generate (adjust as needed).
    """

    async def dispatch(self, request: Request, call_next):
        # Only guard heavy routes (tweak the condition for your API)
        if request.method == "POST" and request.url.path.startswith("/v1/generate"):
            async with _semaphore:
                return await call_next(request)
        else:
            return await call_next(request)


def setup(app) -> None:
    """Install concurrency middleware."""
    app.add_middleware(_ConcurrencyMiddleware)