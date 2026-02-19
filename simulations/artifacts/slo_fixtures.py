# simulations/artifacts/slo_fixtures.py
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from llm_contracts.runtime.generate_slo import RUNTIME_GENERATE_SLO_VERSION

from simulations.artifacts.contracts import write_slo, verify_slo_payload


def _utc_now_iso() -> str:
    # RFC3339-ish, matches schema "date-time"
    return datetime.now(timezone.utc).isoformat()


def build_generate_slo_payload(
    *,
    routes: List[str],
    window_seconds: int,
    model_id: str,
    total_requests: int,
    error_total: int,
    error_rate: float,
    latency_p50_ms: float,
    latency_p95_ms: float,
    latency_p99_ms: float,
    latency_avg_ms: float,
    latency_max_ms: float,
    prompt_tokens_avg: float,
    prompt_tokens_max: int,
    completion_tokens_avg: float,
    completion_tokens_p95: float,
    completion_tokens_max: int,
    generated_at: Optional[str] = None,
    window_end: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a schema-valid runtime_generate_slo_v1 payload.

    This is the minimal structure that:
      - passes JSON Schema
      - supports parse_generate_slo() expectations
    """
    if not routes:
        raise ValueError("routes must be non-empty")

    ts = generated_at or _utc_now_iso()
    we = window_end or ts

    model_row: Dict[str, Any] = {
        "model_id": model_id,
        "requests": {"total": int(total_requests)},
        "errors": {
            "total": int(error_total),
            "rate": float(error_rate),
            "by_status": {},
            "by_code": {},
        },
        "latency_ms": {
            "p50": float(latency_p50_ms),
            "p95": float(latency_p95_ms),
            "p99": float(latency_p99_ms),
            "avg": float(latency_avg_ms),
            "max": float(latency_max_ms),
        },
        "tokens": {
            "prompt": {"avg": float(prompt_tokens_avg), "max": int(prompt_tokens_max)},
            "completion": {
                "avg": float(completion_tokens_avg),
                "p95": float(completion_tokens_p95),
                "max": int(completion_tokens_max),
            },
        },
    }

    payload: Dict[str, Any] = {
        "schema_version": RUNTIME_GENERATE_SLO_VERSION,
        "generated_at": ts,
        "window_seconds": int(window_seconds),
        "window_end": we,
        "routes": list(routes),
        "models": [model_row],
        "totals": {
            "requests": {"total": int(total_requests)},
            "errors": {
                "total": int(error_total),
                "rate": float(error_rate),
                "by_status": {},
                "by_code": {},
            },
            "latency_ms": {
                "p50": float(latency_p50_ms),
                "p95": float(latency_p95_ms),
                "p99": float(latency_p99_ms),
                "avg": float(latency_avg_ms),
                "max": float(latency_max_ms),
            },
            "tokens": {
                "prompt": {"avg": float(prompt_tokens_avg), "max": int(prompt_tokens_max)},
                "completion": {
                    "avg": float(completion_tokens_avg),
                    "p95": float(completion_tokens_p95),
                    "max": int(completion_tokens_max),
                },
            },
        },
    }

    # Local sanity: ensure this payload is parse-valid (optional but useful)
    vr = verify_slo_payload(payload)
    if not vr.ok:
        raise ValueError(f"SLO payload failed verification: {vr.error}")

    return payload


def good_generate_slo_payload(
    *,
    model_id: str = "demo-model",
    routes: Optional[List[str]] = None,
    window_seconds: int = 300,
) -> Dict[str, Any]:
    """
    "Good" preset: low error, low latency, moderate completion tokens.
    """
    return build_generate_slo_payload(
        routes=routes or ["/v1/generate", "/v1/generate/batch"],
        window_seconds=window_seconds,
        model_id=model_id,
        total_requests=200,
        error_total=0,
        error_rate=0.0,
        latency_p50_ms=80.0,
        latency_p95_ms=180.0,
        latency_p99_ms=260.0,
        latency_avg_ms=95.0,
        latency_max_ms=320.0,
        prompt_tokens_avg=220.0,
        prompt_tokens_max=600,
        completion_tokens_avg=180.0,
        completion_tokens_p95=320.0,
        completion_tokens_max=512,
    )


def bad_generate_slo_payload(
    *,
    model_id: str = "demo-model",
    routes: Optional[List[str]] = None,
    window_seconds: int = 300,
) -> Dict[str, Any]:
    """
    "Bad" preset: non-trivial error rate and large completion tokens p95.
    This is the kind of snapshot that should push the policy toward clamping.
    """
    return build_generate_slo_payload(
        routes=routes or ["/v1/generate", "/v1/generate/batch"],
        window_seconds=window_seconds,
        model_id=model_id,
        total_requests=200,
        error_total=20,
        error_rate=0.10,
        latency_p50_ms=250.0,
        latency_p95_ms=1400.0,
        latency_p99_ms=2200.0,
        latency_avg_ms=520.0,
        latency_max_ms=4000.0,
        prompt_tokens_avg=260.0,
        prompt_tokens_max=900,
        completion_tokens_avg=900.0,
        completion_tokens_p95=1800.0,
        completion_tokens_max=4096,
    )


def write_generate_slo_latest(
    out_path: Path,
    payload: Dict[str, Any],
) -> Path:
    """
    Write the payload to a canonical 'latest.json' location.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return write_slo(out_path, payload)


def write_generate_slo_latest_under_repo(
    repo_root: Path,
    payload: Dict[str, Any],
) -> Path:
    """
    Convenience for the canonical location:
      slo_out/generate/latest.json
    """
    return write_generate_slo_latest(repo_root / "slo_out" / "generate" / "latest.json", payload)