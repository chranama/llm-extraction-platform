# contracts/src/llm_contracts/runtime/generate_slo.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast

from llm_contracts.schema import atomic_write_json_internal, validate_internal

Pathish = Union[str, Path]

RUNTIME_GENERATE_SLO_SCHEMA = "runtime_generate_slo_v1.schema.json"
RUNTIME_GENERATE_SLO_VERSION = "runtime_generate_slo_v1"


@dataclass(frozen=True)
class GenerateSLOSnapshot:
    """
    Stable minimal snapshot policy can consume.
    Keep `raw` for forward compatibility.
    """

    schema_version: str
    generated_at: str
    window_seconds: int
    window_end: str
    routes: List[str]

    # totals (these are what policy typically gates on)
    total_requests: int
    error_rate: float
    latency_p95_ms: float
    completion_tokens_p95: float

    raw: Dict[str, Any]
    source_path: Optional[str] = None
    error: Optional[str] = None


def parse_generate_slo(payload: Dict[str, Any], *, source_path: Optional[str] = None) -> GenerateSLOSnapshot:
    validate_internal(RUNTIME_GENERATE_SLO_SCHEMA, payload)

    schema_version = str(payload["schema_version"]).strip()
    if schema_version != RUNTIME_GENERATE_SLO_VERSION:
        raise ValueError(f"Unsupported generate_slo schema_version: {schema_version}")

    totals = cast(Dict[str, Any], payload.get("totals") or {})
    req = cast(Dict[str, Any], totals.get("requests") or {})
    err = cast(Dict[str, Any], totals.get("errors") or {})
    lat = cast(Dict[str, Any], totals.get("latency_ms") or {})
    tok = cast(Dict[str, Any], totals.get("tokens") or {})
    comp = cast(Dict[str, Any], tok.get("completion") or {})

    total_requests = int(req.get("total") or 0)
    error_rate = float(err.get("rate") or 0.0)
    latency_p95_ms = float(lat.get("p95") or 0.0)
    completion_tokens_p95 = float(comp.get("p95") or 0.0)

    return GenerateSLOSnapshot(
        schema_version=schema_version,
        generated_at=str(payload["generated_at"]).strip(),
        window_seconds=int(payload["window_seconds"]),
        window_end=str(payload["window_end"]).strip(),
        routes=list(payload.get("routes") or []),
        total_requests=total_requests,
        error_rate=error_rate,
        latency_p95_ms=latency_p95_ms,
        completion_tokens_p95=completion_tokens_p95,
        raw=dict(payload),
        source_path=source_path,
        error=None,
    )


def read_generate_slo(path: Pathish) -> GenerateSLOSnapshot:
    p = Path(path).resolve()
    try:
        import json

        payload = cast(Dict[str, Any], json.loads(p.read_text(encoding="utf-8")))
        return parse_generate_slo(payload, source_path=str(p))
    except Exception as e:
        return GenerateSLOSnapshot(
            schema_version="",
            generated_at="",
            window_seconds=0,
            window_end="",
            routes=[],
            total_requests=0,
            error_rate=1.0,
            latency_p95_ms=0.0,
            completion_tokens_p95=0.0,
            raw={},
            source_path=str(p),
            error=f"generate_slo_parse_error: {type(e).__name__}: {e}",
        )


def write_generate_slo(path: Pathish, payload: Dict[str, Any]) -> Path:
    return atomic_write_json_internal(RUNTIME_GENERATE_SLO_SCHEMA, path, payload)