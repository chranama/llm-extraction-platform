# server/src/llm_server/cli.py
from __future__ import annotations

import os
from typing import Optional

import typer
import uvicorn

app = typer.Typer(
    name="llm",
    help="Unified CLI for llm-server (serve).",
    no_args_is_help=True,
)


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------
@app.command("serve")
def serve(
    host: str = typer.Option("0.0.0.0", "--host", envvar="HOST", help="Bind host"),
    port: int = typer.Option(8000, "--port", envvar="PORT", help="Bind port"),
    workers: int = typer.Option(1, "--workers", envvar="WORKERS", help="Uvicorn workers (prod)"),
    reload_: Optional[bool] = typer.Option(
        None,
        "--reload/--no-reload",
        help="Enable auto-reload (overrides UVICORN_RELOAD if provided).",
    ),
    proxy_headers: bool = typer.Option(True, "--proxy-headers/--no-proxy-headers", help="Respect proxy headers"),
    profile: Optional[str] = typer.Option(
        None,
        "--profile",
        envvar="APP_PROFILE",
        help="Config profile inside config/server.yaml (e.g. host, docker).",
    ),
):
    """
    Run the FastAPI service.
    """
    if profile:
        os.environ["APP_PROFILE"] = profile

    env = os.getenv("ENV", "").lower()
    dev_mode = env == "dev" or os.getenv("DEV") == "1"

    if reload_ is None:
        reload_enabled = _env_flag("UVICORN_RELOAD", "0")
    else:
        reload_enabled = bool(reload_)

    if dev_mode:
        uvicorn.run(
            "llm_server.main:create_app",
            factory=True,
            host=host,
            port=port,
            reload=reload_enabled,
            workers=1,
            proxy_headers=proxy_headers,
        )
        return

    uvicorn.run(
        "llm_server.main:create_app",
        factory=True,
        host=host,
        port=port,
        workers=int(workers),
        proxy_headers=proxy_headers,
    )


def main() -> None:
    app()