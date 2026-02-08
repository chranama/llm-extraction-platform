# server/src/llm_server/io/policy_decisions.py
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from llm_contracts.runtime.policy_decision import (
    PolicyDecisionSnapshot as ContractsPolicyDecisionSnapshot,
    read_policy_decision,
)


@dataclass(frozen=True)
class PolicyDecisionSnapshot:
    """
    Backend-local minimal runtime representation.

    Design goals:
      - Small, explicit, stable
      - Fail-closed for gating
      - Shaping is advisory only
    """
    ok: bool

    # Scope
    model_id: Optional[str]

    # Gating
    enable_extract: Optional[bool]

    # Runtime shaping (v2)
    generate_max_new_tokens_cap: Optional[int]

    # Debug / forward-compat
    raw: Dict[str, Any]
    source_path: Optional[str]
    error: Optional[str]


# ------------------------------------------------------------------------------
# Conversion
# ------------------------------------------------------------------------------


def _to_backend_snapshot(
    s: ContractsPolicyDecisionSnapshot,
) -> PolicyDecisionSnapshot:
    """
    Convert contracts snapshot -> backend snapshot.

    Backend semantics:
      - ok=False => extract must be disabled (fail-closed)
      - enable_extract=None => no override
      - generate_max_new_tokens_cap is advisory only
    """

    # Defensive normalization of cap
    cap = None
    raw_cap = getattr(s, "generate_max_new_tokens_cap", None)
    if isinstance(raw_cap, int) and raw_cap > 0:
        cap = raw_cap

    return PolicyDecisionSnapshot(
        ok=bool(s.ok),
        model_id=s.model_id,
        enable_extract=(
            bool(s.enable_extract)
            if s.ok and s.enable_extract is not None
            else False if not s.ok else None
        ),
        generate_max_new_tokens_cap=cap,
        raw=dict(s.raw or {}),
        source_path=s.source_path,
        error=s.error,
    )


# ------------------------------------------------------------------------------
# Loading / caching
# ------------------------------------------------------------------------------


def load_policy_decision_from_env() -> PolicyDecisionSnapshot:
    """
    Load a policy decision JSON from POLICY_DECISION_PATH.

    Semantics:
      - POLICY_DECISION_PATH unset:
          -> no override (ok=True, enable_extract=None)
      - path set but missing:
          -> fail-closed (ok=False, enable_extract=False)
      - parse/validation failure:
          -> fail-closed
      - parsed but decision.ok == False:
          -> fail-closed
    """
    path_s = os.getenv("POLICY_DECISION_PATH", "").strip()
    if not path_s:
        return PolicyDecisionSnapshot(
            ok=True,
            model_id=None,
            enable_extract=None,
            generate_max_new_tokens_cap=None,
            raw={},
            source_path=None,
            error=None,
        )

    p = Path(path_s)
    if not p.exists():
        return PolicyDecisionSnapshot(
            ok=False,
            model_id=None,
            enable_extract=False,
            generate_max_new_tokens_cap=None,
            raw={},
            source_path=str(p),
            error="policy_decision_missing",
        )

    try:
        snap = read_policy_decision(p)
        return _to_backend_snapshot(snap)
    except Exception as e:
        return PolicyDecisionSnapshot(
            ok=False,
            model_id=None,
            enable_extract=False,
            generate_max_new_tokens_cap=None,
            raw={},
            source_path=str(p),
            error=f"policy_decision_read_error: {type(e).__name__}: {e}",
        )


def get_policy_snapshot(request) -> PolicyDecisionSnapshot:
    """
    Get cached snapshot from app.state if present; else load and cache.
    """
    snap = getattr(request.app.state, "policy_snapshot", None)
    if isinstance(snap, PolicyDecisionSnapshot):
        return snap

    snap = load_policy_decision_from_env()
    request.app.state.policy_snapshot = snap
    return snap


def reload_policy_snapshot(request) -> PolicyDecisionSnapshot:
    """
    Force reload from disk and overwrite app.state cache.
    """
    snap = load_policy_decision_from_env()
    request.app.state.policy_snapshot = snap
    return snap


# ------------------------------------------------------------------------------
# Application helpers
# ------------------------------------------------------------------------------


def policy_capability_overrides(
    model_id: str, *, request
) -> Optional[Dict[str, bool]]:
    """
    Capability overrides for models.yaml semantics.

    Rules:
      - If policy file configured AND ok=False => extract disabled for ALL models
      - If decision scoped to model_id and does not match => no override
      - If enable_extract is None => no override
    """
    snap = get_policy_snapshot(request)

    # Policy configured but non-ok => fail-closed globally
    if snap.source_path and not snap.ok:
        return {"extract": False}

    # Scoped decision mismatch
    if snap.model_id and snap.model_id != model_id:
        return None

    if snap.enable_extract is None:
        return None

    return {"extract": bool(snap.enable_extract)}


def policy_generate_max_new_tokens_cap(
    model_id: str, *, request
) -> Optional[int]:
    """
    Optional runtime shaping for /v1/generate.

    Rules:
      - Never invent a cap
      - Scoped by model_id if present
      - ok=False does NOT imply a cap
    """
    snap = get_policy_snapshot(request)

    if snap.model_id and snap.model_id != model_id:
        return None

    cap = snap.generate_max_new_tokens_cap
    if isinstance(cap, int) and cap > 0:
        return cap

    return None