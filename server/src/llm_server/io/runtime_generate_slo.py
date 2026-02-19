# server/src/llm_server/io/runtime_generate_slo.py
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from sqlalchemy.ext.asyncio import AsyncSession

from llm_contracts.runtime.generate_slo import parse_generate_slo
from llm_server.telemetry.queries import compute_window_generate_slo_contracts_payload


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
      - SLO_OUT_DIR can override root directory
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
    Compute SLO snapshot from DB logs (InferenceLog) and write a CONTRACTS-SHAPED
    runtime_generate_slo_v1 artifact.

    This MUST remain aligned with:
      - llm_contracts.runtime.generate_slo.read_generate_slo()
      - policy/src/llm_policy/io/generate_slo.py (reads contracts snapshot)
    """
    payload = await compute_window_generate_slo_contracts_payload(
        session,
        window_seconds=int(window_seconds),
        routes=list(routes) if routes else None,
        model_id=model_id,
    )

    # Validate at the boundary so admin endpoints are "true" boundaries.
    # parse_generate_slo() runs JSON Schema validation + version checks.
    parse_generate_slo(dict(payload))

    path = Path(out_path) if out_path is not None else default_generate_slo_path()
    _atomic_write_json(path, payload)
    return SloWriteResult(ok=True, out_path=str(path), payload=dict(payload))


# -----------------------------
# Internals
# -----------------------------


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """
    Atomically write JSON to disk (write temp file in same dir, fsync, then os.replace).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n"

    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            delete=False,
            dir=str(path.parent),
            prefix=f".{path.name}.",
            suffix=".tmp",
        ) as f:
            tmp_path = Path(f.name)
            f.write(data)
            f.flush()
            os.fsync(f.fileno())

        os.replace(str(tmp_path), str(path))
    finally:
        if tmp_path is not None and tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass