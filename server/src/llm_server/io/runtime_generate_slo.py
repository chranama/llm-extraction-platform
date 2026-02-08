# server/src/llm_server/io/runtime_generate_slo.py
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from sqlalchemy.ext.asyncio import AsyncSession

from llm_server.telemetry.queries import compute_window_slo_snapshot


# -----------------------------
# Public API
# -----------------------------


@dataclass(frozen=True)
class SloWriteResult:
    ok: bool
    out_path: str
    payload: dict[str, Any]


def default_generate_slo_dir() -> Path:
    """
    Convention:
      - SLO_OUT_DIR can override root
      - default is repo-relative "slo_out/generate"
    """
    root = os.getenv("SLO_OUT_DIR", "").strip()
    if root:
        return Path(root)
    return Path("slo_out") / "generate"


def default_generate_slo_path(*, filename: str = "latest.json") -> Path:
    return default_generate_slo_dir() / filename


async def write_generate_slo_artifact(
    session: AsyncSession,
    *,
    window_seconds: int,
    routes: Sequence[str] | None = None,
    model_id: str | None = None,
    out_path: str | os.PathLike[str] | None = None,
) -> SloWriteResult:
    """
    Compute SLO snapshot from DB logs (InferenceLog) and write an artifact.

    Artifact shape:
      {
        "schema_version": "runtime_generate_slo_v1",
        "generated_at": "...",
        "window_seconds": ...,
        "window_end": "...",
        "since": "...",
        "routes": [...],
        "model_id": "...|null",
        "totals": { "requests": ..., "errors": ..., "error_rate": ... },
        "latency_ms": { "avg": ..., "p95": ... },
        "tokens": { "prompt_total": ..., "completion_total": ..., "completion_p95": ... }
      }

    Returns payload + output path.
    """
    snap = await compute_window_slo_snapshot(
        session,
        window_seconds=int(window_seconds),
        routes=list(routes) if routes else None,
        model_id=model_id,
    )

    # Normalize timestamps as ISO strings for the artifact boundary.
    payload = _to_jsonable(
        {
            "schema_version": "runtime_generate_slo_v1",
            "generated_at": datetime.utcnow().isoformat() + "Z",
            **snap,
        }
    )

    path = Path(out_path) if out_path is not None else default_generate_slo_path()
    _atomic_write_json(path, payload)

    return SloWriteResult(ok=True, out_path=str(path), payload=payload)


# -----------------------------
# Internals
# -----------------------------


def _to_jsonable(x: Any) -> Any:
    if isinstance(x, datetime):
        # keep timezone info if present
        return x.isoformat()
    if isinstance(x, dict):
        return {k: _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, tuple):
        return [_to_jsonable(v) for v in x]
    return x


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n"

    # Atomic-ish write: write to temp in same directory, then replace.
    tmp_fd = None
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            delete=False,
            dir=str(path.parent),
            prefix=f".{path.name}.",
            suffix=".tmp",
        ) as f:
            tmp_fd = f.fileno()
            tmp_path = Path(f.name)
            f.write(data)
            f.flush()
            os.fsync(tmp_fd)

        os.replace(str(tmp_path), str(path))
    finally:
        # Clean up temp file if anything failed before replace.
        if tmp_path is not None and tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass