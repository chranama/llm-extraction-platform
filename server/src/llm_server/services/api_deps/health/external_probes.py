# server/src/llm_server/services/api_deps/health/external_probes.py
from __future__ import annotations

from typing import Any, Dict, Tuple

import anyio


def sync_llamacpp_dependency_check(backend_obj: Any) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Synchronous readiness check for llama-server.
    This should be fast (GET /health).
    """
    try:
        fn = getattr(backend_obj, "is_ready", None)
        if callable(fn):
            ok, details = fn()
            okb = bool(ok)
            return okb, ("ok" if okb else "not ready"), (details if isinstance(details, dict) else {"details": details})

        client = getattr(backend_obj, "_client", None)
        health_fn = getattr(client, "health", None) if client is not None else None
        if callable(health_fn):
            data = health_fn()
            okb = bool(isinstance(data, dict) and data.get("status") == "ok")
            return okb, ("ok" if okb else "not ready"), {"health": data}

        return False, "missing health check", {"reason": "llamacpp backend lacks is_ready() / client.health()"}
    except Exception as e:
        return False, "error", {"error": repr(e)}


def sync_external_backend_generate_check(backend_obj: Any) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Best-effort check to prove an external backend can generate WITHOUT loading weights locally.
    Prefers can_generate(), falls back to is_ready().
    """
    try:
        fn = getattr(backend_obj, "can_generate", None)
        if callable(fn):
            ok, details = fn()
            okb = bool(ok)
            return okb, ("ok" if okb else "not ready"), (details if isinstance(details, dict) else {"details": details})

        fn2 = getattr(backend_obj, "is_ready", None)
        if callable(fn2):
            ok, details = fn2()
            okb = bool(ok)
            return okb, ("ok" if okb else "not ready"), (
                (details if isinstance(details, dict) else {"details": details}) | {"note": "can_generate missing; used is_ready"}
            )

        return False, "missing readiness probe", {"reason": "backend missing can_generate() and is_ready()"}
    except Exception as e:
        return False, "error", {"error": repr(e)}


def sync_remote_probe(backend_obj: Any) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Remote backend "probe" mode.
    Prefer backend.is_ready(); else try client.health().
    """
    try:
        fn = getattr(backend_obj, "is_ready", None)
        if callable(fn):
            ok, details = fn()
            okb = bool(ok)
            return okb, ("ok" if okb else "not ready"), (details if isinstance(details, dict) else {"details": details})

        client = getattr(backend_obj, "_client", None)
        health_fn = getattr(client, "health", None) if client is not None else None
        if callable(health_fn):
            data = health_fn()
            okb = bool(
                isinstance(data, dict)
                and (
                    data.get("status") == "ok"
                    or data.get("ok") is True
                    or data.get("healthy") is True
                )
            )
            return okb, ("ok" if okb else "not ready"), {"health": data}

        return False, "missing remote probe", {"reason": "remote backend lacks is_ready() and client.health()"}
    except Exception as e:
        return False, "error", {"error": repr(e)}


async def llamacpp_dependency_check_async(backend_obj: Any) -> Tuple[bool, str, Dict[str, Any]]:
    return await anyio.to_thread.run_sync(sync_llamacpp_dependency_check, backend_obj)


async def external_backend_generate_check_async(backend_obj: Any) -> Tuple[bool, str, Dict[str, Any]]:
    return await anyio.to_thread.run_sync(sync_external_backend_generate_check, backend_obj)


async def remote_probe_async(backend_obj: Any) -> Tuple[bool, str, Dict[str, Any]]:
    return await anyio.to_thread.run_sync(sync_remote_probe, backend_obj)