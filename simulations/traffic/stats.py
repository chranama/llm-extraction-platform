# simulations/traffic/stats.py
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Optional, Sequence

from simulations.traffic.models import Event, Summary


def _percentile(sorted_vals: Sequence[float], p: float) -> Optional[float]:
    if not sorted_vals:
        return None
    if p <= 0:
        return float(sorted_vals[0])
    if p >= 100:
        return float(sorted_vals[-1])
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(sorted_vals[int(k)])
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return float(d0 + d1)


def _is_timeout_like(e: Event) -> bool:
    """
    Best-effort: simulations.traffic.client uses urllib.
    Typical timeout-ish failures show up as:
      - status is None (URLError / socket timeout)
      - error string contains timeout-ish tokens
    """
    if e.ok:
        return False
    msg = (e.error or "").lower()
    if "timed out" in msg or "timeout" in msg:
        return True
    return False


def _is_conn_error(e: Event) -> bool:
    """
    Connection failures in SimClientError show up as status=None.
    """
    return (not e.ok) and (e.status is None)


def compute_summary(
    *,
    run_id: str,
    scenario: str,
    duration_s: float,
    events: Iterable[Event],
) -> Summary:
    items = list(events)
    sent = len(items)
    completed = len(items)

    ok_count = sum(1 for e in items if e.ok)
    err_count = sent - ok_count
    error_rate = (err_count / sent) if sent > 0 else 0.0

    # latency (all)
    lat_all = sorted([float(e.elapsed_ms) for e in items if e.elapsed_ms is not None])
    avg_latency_all = (sum(lat_all) / len(lat_all)) if lat_all else None

    # latency (success-only) — this is usually what you care about
    lat_ok = sorted([float(e.elapsed_ms) for e in items if e.ok and e.elapsed_ms is not None])
    avg_latency_ok = (sum(lat_ok) / len(lat_ok)) if lat_ok else None

    cached_vals = [e.cached for e in items if e.cached is not None]
    cache_hit_rate = None
    if cached_vals:
        cache_hit_rate = sum(1 for x in cached_vals if x) / len(cached_vals)

    pt = [e.prompt_tokens for e in items if isinstance(e.prompt_tokens, int)]
    ct = [e.completion_tokens for e in items if isinstance(e.completion_tokens, int)]
    total_prompt_tokens = sum(pt) if pt else None
    total_completion_tokens = sum(ct) if ct else None

    # basic breakdowns
    by_status: Dict[str, int] = {}
    for e in items:
        k = str(e.status) if e.status is not None else ("ok" if e.ok else "error")
        by_status[k] = by_status.get(k, 0) + 1

    by_endpoint: Dict[str, int] = {}
    for e in items:
        by_endpoint[e.endpoint] = by_endpoint.get(e.endpoint, 0) + 1

    # gate/overload focused breakdowns
    n_429 = sum(1 for e in items if (e.status == 429))
    n_5xx = sum(1 for e in items if isinstance(e.status, int) and 500 <= e.status <= 599)
    n_conn = sum(1 for e in items if _is_conn_error(e))
    n_timeout_like = sum(1 for e in items if _is_timeout_like(e))

    gate_breakdown: Dict[str, Any] = {
        "count_429": int(n_429),
        "rate_429": (float(n_429) / sent) if sent > 0 else 0.0,
        "count_5xx": int(n_5xx),
        "rate_5xx": (float(n_5xx) / sent) if sent > 0 else 0.0,
        "count_conn_errors": int(n_conn),
        "rate_conn_errors": (float(n_conn) / sent) if sent > 0 else 0.0,
        "count_timeout_like": int(n_timeout_like),
        "rate_timeout_like": (float(n_timeout_like) / sent) if sent > 0 else 0.0,
    }

    return Summary(
        run_id=run_id,
        scenario=scenario,
        duration_s=float(duration_s),
        sent=sent,
        completed=completed,
        # IMPORTANT: this "ok" is NOT meaningful for overload tests,
        # but we keep it for backward compatibility.
        ok=(err_count == 0),
        error_rate=float(error_rate),

        # Keep original latency fields (all requests)
        avg_latency_ms=avg_latency_all,
        latency_p50_ms=_percentile(lat_all, 50),
        latency_p95_ms=_percentile(lat_all, 95),
        latency_p99_ms=_percentile(lat_all, 99),

        cache_hit_rate=cache_hit_rate,
        total_prompt_tokens=total_prompt_tokens,
        total_completion_tokens=total_completion_tokens,
        breakdown={
            "by_status": by_status,
            "by_endpoint": by_endpoint,
            "gate": gate_breakdown,

            # Bonus: expose success-only latency as extra fields (non-breaking; Summary allows dict only, so put here)
            "latency_ok": {
                "avg_latency_ms": avg_latency_ok,
                "p50_ms": _percentile(lat_ok, 50),
                "p95_ms": _percentile(lat_ok, 95),
                "p99_ms": _percentile(lat_ok, 99),
                "n_ok": int(len(lat_ok)),
            },
        },
    )