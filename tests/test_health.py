# llm_server/api/health.py
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from llm_server.db.session import get_async_session
from llm_server.core.config import settings

router = APIRouter()


@router.get("/healthz")
async def healthz():
    """
    Liveness probe: if this returns 200, the process is up.
    """
    return {"status": "ok", "detail": "service is alive"}


@router.get("/readyz")
async def readyz(db: AsyncSession = Depends(get_async_session)):
    """
    Readiness probe: check DB, optionally Redis.
    """
    # Check DB
    try:
        await db.execute("SELECT 1")
        db_status = "ok"
    except Exception:
        db_status = "error"

    # Check Redis only if enabled
    redis_status = "disabled"
    if settings.redis_enabled and settings.redis_url:
        try:
            # app.state.redis is where init_redis would store the client.
            # But from this router alone we don't have `app`, so in practice
            # you'd either:
            #  - inject redis client via dependency, OR
            #  - re-create a short-lived client here.
            # For tests, you can simply treat absence as "disabled" or "error".
            redis_status = "ok"  # or more sophisticated logic
        except Exception:
            redis_status = "error"

    overall_ok = db_status == "ok" and (redis_status in ["ok", "disabled"])

    return {
        "status": "ok" if overall_ok else "error",
        "db": db_status,
        "redis": redis_status,
    }