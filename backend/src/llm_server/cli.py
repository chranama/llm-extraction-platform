# src/llm_server/cli.py
import os
import uvicorn

def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}

def serve():
    env = os.getenv("ENV", "").lower()
    dev_mode = env == "dev" or os.getenv("DEV") == "1"

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))

    # IMPORTANT: reload should be opt-in, not automatic in dev
    reload_enabled = _env_flag("UVICORN_RELOAD", "0")

    if dev_mode:
        uvicorn.run(
            "llm_server.main:create_app",
            factory=True,
            host=host,
            port=port,
            reload=reload_enabled,
            # also force single worker in dev; stateful in-memory LLM needs this
            workers=1,
            proxy_headers=True,
        )
        return

    workers = int(os.getenv("WORKERS", "1"))
    uvicorn.run(
        "llm_server.main:create_app",
        factory=True,
        host=host,
        port=port,
        workers=workers,
        proxy_headers=True,
    )