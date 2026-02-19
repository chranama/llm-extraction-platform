# server/src/llm_server/io/policy_decisions.py
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from llm_contracts.runtime.policy_decision import (
    PolicyDecisionSnapshot as ContractsPolicyDecisionSnapshot,
    read_policy_decision,
)

logger = logging.getLogger("llm.policy")


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


def _to_backend_snapshot(s: ContractsPolicyDecisionSnapshot) -> PolicyDecisionSnapshot:
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
        ok=bool(getattr(s, "ok", False)),
        model_id=getattr(s, "model_id", None),
        enable_extract=(
            bool(getattr(s, "enable_extract", False))
            if getattr(s, "ok", False) and getattr(s, "enable_extract", None) is not None
            else False if not getattr(s, "ok", False) else None
        ),
        generate_max_new_tokens_cap=cap,
        raw=dict(getattr(s, "raw", None) or {}),
        source_path=getattr(s, "source_path", None),
        error=getattr(s, "error", None),
    )


# ------------------------------------------------------------------------------
# Loading / caching
# ------------------------------------------------------------------------------


def _best_effort_parse_v2_json(p: Path) -> Optional[PolicyDecisionSnapshot]:
    """
    Fallback parser for policy_decision_v2 JSON.

    Why this exists:
      - If llm_contracts is behind (doesn't recognize v2 yet), read_policy_decision()
        can throw and the backend ends up fail-closed with raw={}
      - This keeps fail-closed semantics for gating, but allows v2 shaping + gating
        to work immediately.

    Notes:
      - We intentionally keep validation minimal (backend is not the contracts authority)
      - Any parse error returns None and caller handles fail-closed
    """
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

    if not isinstance(obj, dict):
        return None

    sv = obj.get("schema_version")
    if sv != "policy_decision_v2":
        return None

    # Minimal field extraction with defensive typing
    ok = bool(obj.get("ok", False))
    model_id = obj.get("policy", {}).get("model_id") if isinstance(obj.get("policy"), dict) else obj.get("model_id")
    if model_id is not None and not isinstance(model_id, str):
        model_id = None

    enable_extract = obj.get("enable_extract", None)
    if enable_extract is not None:
        enable_extract = bool(enable_extract)

    cap = obj.get("generate_max_new_tokens_cap", None)
    if not (isinstance(cap, int) and cap > 0):
        cap = None

    # Fail-closed gating semantics: if ok is False, extract must be disabled
    eff_enable_extract: Optional[bool]
    if not ok:
        eff_enable_extract = False
    else:
        # ok=True:
        # - enable_extract=None => no override
        # - enable_extract=True/False => explicit override
        eff_enable_extract = enable_extract if enable_extract is not None else None

    return PolicyDecisionSnapshot(
        ok=ok,
        model_id=model_id,
        enable_extract=eff_enable_extract,
        generate_max_new_tokens_cap=cap,
        raw=obj,
        source_path=str(p),
        error=None,
    )


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

    # First try contracts parser (authoritative when it works)
    try:
        snap = read_policy_decision(p)
        return _to_backend_snapshot(snap)
    except Exception as e:
        # Fallback: attempt to parse v2 JSON directly so policy works even if contracts lags
        fb = _best_effort_parse_v2_json(p)
        if fb is not None:
            logger.warning(
                "read_policy_decision failed; using best-effort v2 parser",
                extra={"path": str(p), "error_type": type(e).__name__},
            )
            return fb

        # Fail closed, but surface a more specific error string
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


def policy_capability_overrides(model_id: str, *, request) -> Optional[Dict[str, bool]]:
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


def policy_generate_max_new_tokens_cap(model_id: str, *, request) -> Optional[int]:
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