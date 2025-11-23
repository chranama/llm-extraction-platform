# app/core/metrics.py
import time
from fastapi import APIRouter, Request, Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# Prometheus metrics
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "Latency of HTTP requests",
    ["method", "endpoint"],
)

# Expose /metrics
router = APIRouter()

@router.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def setup(app) -> None:
    """Add latency/count middleware and mount /metrics route."""
    @app.middleware("http")
    async def prometheus_metrics(request: Request, call_next):
        start = time.time()
        response: Response = await call_next(request)
        duration = time.time() - start

        endpoint = request.url.path
        method = request.method
        status = response.status_code

        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status).inc()
        REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(duration)

        return response

    app.include_router(router)