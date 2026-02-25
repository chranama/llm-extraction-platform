# server/src/llm_server/services/api_deps/generate/generate_policy.py
from __future__ import annotations

from typing import Any, Optional, Tuple

from fastapi import Request

from llm_server.io.policy_decisions import get_policy_snapshot
from llm_server.services.api_deps.core.policy_snapshot import snapshot_generate_cap


def normalize_positive_int(x: Any) -> Optional[int]:
    """
    Normalize an optional "positive int" input.

    - None -> None
    - bool -> None  (avoid True/False becoming 1/0)
    - non-int-like -> None
    - <= 0 -> None
    - > 0 -> int
    """
    if x is None or isinstance(x, bool):
        return None
    try:
        i = int(x)
    except Exception:
        return None
    return i if i > 0 else None


def apply_generate_cap(
    request: Request,
    *,
    model_id: str,
    requested_max_new_tokens: int | None,
) -> Tuple[int | None, int | None, bool]:
    """
    Applies policy-controlled cap to requested max_new_tokens.

    Returns:
      (effective_max_new_tokens, policy_cap, clamped)

    Semantics:
      - If no policy cap => effective=requested, policy_cap=None, clamped=False
      - If policy cap exists and requested is None => effective=cap, policy_cap=cap, clamped=False
      - If both exist => effective=min(requested, cap), clamped=(requested > cap)

    NOTE:
      - Policy is sourced from the request-scoped policy snapshot (cached in app.state).
      - This keeps admin reload + health/admin inspection consistent.
    """
    # Best-effort: policy snapshot may be missing/unavailable; treat as no cap.
    try:
        snap = get_policy_snapshot(request)
    except Exception:
        snap = None

    cap_raw = snapshot_generate_cap(snap) if snap is not None else None
    cap_i = normalize_positive_int(cap_raw)
    req_i = normalize_positive_int(requested_max_new_tokens)

    if cap_i is None:
        return requested_max_new_tokens, None, False

    if req_i is None:
        return cap_i, cap_i, False

    eff = min(req_i, cap_i)
    clamped = req_i > cap_i
    return eff, cap_i, clamped