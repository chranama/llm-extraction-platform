# policy/src/llm_policy/io/generate_slo.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

from llm_contracts.runtime.generate_slo import GenerateSLOSnapshot, read_generate_slo
from llm_policy.types.artifact_read import ArtifactReadResult

Pathish = Union[str, Path]


def default_generate_slo_path() -> Path:
    """
    Convention (match server):
      - if SLO_OUT_DIR is set, read <SLO_OUT_DIR>/latest.json
      - otherwise default to repo-relative slo_out/generate/latest.json

    Must stay aligned with:
      server/src/llm_server/io/runtime_generate_slo.py
        - default_generate_slo_dir()
        - default_generate_slo_path()
    """
    root = os.getenv("SLO_OUT_DIR", "").strip()
    if root:
        return Path(root) / "latest.json"
    return Path("slo_out") / "generate" / "latest.json"


def resolve_generate_slo_path(path: Optional[Pathish] = None) -> Path:
    """
    Resolve the path policy should read.

    If path is None/empty:
      - uses default_generate_slo_path()

    If path is provided:
      - treats it as a direct file path (not a directory)
    """
    if path is None:
        return default_generate_slo_path()

    p = str(path).strip()
    if not p:
        return default_generate_slo_path()

    pp = Path(p)
    # Guardrail: require a filename
    if pp.name in ("", ".", ".."):
        raise ValueError(f"generate_slo path must be a file path, got: {path!r}")
    return pp


def read_generate_slo_snapshot(path: Optional[Pathish] = None) -> GenerateSLOSnapshot:
    """
    Read and validate the generate SLO snapshot artifact.

    Fail-closed behavior:
      - On any read/parse/validate error, contracts.read_generate_slo() returns a
        GenerateSLOSnapshot with:
          - error != None
          - error_rate = 1.0
          - other numeric fields defaulted
    """
    p = resolve_generate_slo_path(path)
    return read_generate_slo(p)


def read_generate_slo_snapshot_result(path: Optional[Pathish] = None) -> ArtifactReadResult[GenerateSLOSnapshot]:
    """
    Same as read_generate_slo_snapshot(), but returns a generic status wrapper
    so callers can log/diagnose without duplicating logic.
    """
    p = resolve_generate_slo_path(path)
    snap = read_generate_slo(p)

    ok = (snap.error is None) and bool(snap.schema_version)
    err = snap.error or None

    return ArtifactReadResult(
        ok=ok,
        artifact=snap,
        resolved_path=str(p),
        error=err,
    )