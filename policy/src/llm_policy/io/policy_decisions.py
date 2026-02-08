# policy/src/llm_policy/io/policy_decisions.py
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Union

from llm_contracts.runtime.policy_decision import (
    PolicyDecisionSnapshot,
    read_policy_decision,
    write_policy_decision,
)
from llm_contracts.schema import validate_internal
from llm_policy.types.decision import Decision

Pathish = Union[str, Path]

POLICY_DECISION_SCHEMA = "policy_decision_v2.schema.json"


def _utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _status_value(x: Any) -> str:
    if x is None:
        return "unknown"
    v = getattr(x, "value", None)
    if isinstance(v, str):
        return v
    if isinstance(x, str):
        return x
    return str(x)


def _issue_to_dict(x: Any) -> dict[str, Any]:
    d: dict[str, Any] = {}
    try:
        d = dict(x.model_dump())
    except Exception:
        if isinstance(x, dict):
            d = dict(x)

    if not d:
        for k in ("code", "message", "context"):
            if hasattr(x, k):
                d[k] = getattr(x, k)

    d.setdefault("code", "issue")
    d.setdefault("message", "")
    d.setdefault("context", {})
    if not isinstance(d["context"], dict):
        d["context"] = {}
    return d


def _decision_to_payload(decision: Decision) -> Dict[str, Any]:
    """
    Convert an in-memory Decision into a policy_decision_v2 artifact payload.
    """
    ok = bool(decision.ok()) if hasattr(decision, "ok") else False

    payload: Dict[str, Any] = {
        "schema_version": "policy_decision_v2",
        "generated_at": _utc_now_iso(),

        # Identity
        "policy": str(getattr(decision, "policy", "") or "").strip() or "unknown_policy",
        "pipeline": str(getattr(decision, "pipeline", "") or "").strip(),

        # Outcome
        "status": _status_value(getattr(decision, "status", None)),
        "ok": ok,

        # Actions / shaping
        "enable_extract": bool(getattr(decision, "enable_extract", False)) if ok else False,
        "generate_max_new_tokens_cap": getattr(
            decision, "generate_max_new_tokens_cap", None
        ),

        # Contract health
        "contract_errors": int(getattr(decision, "contract_errors", 0) or 0),
        "contract_warnings": int(getattr(decision, "contract_warnings", 0) or 0),

        # Provenance
        "thresholds_profile": getattr(decision, "thresholds_profile", None),
        "thresholds_version": getattr(decision, "thresholds_version", None),
        "generate_thresholds_profile": getattr(decision, "generate_thresholds_profile", None),
        "eval_run_dir": getattr(decision, "eval_run_dir", None),
        "eval_task": getattr(decision, "eval_task", None),
        "eval_run_id": getattr(decision, "eval_run_id", None),
        "model_id": getattr(decision, "model_id", None),

        # Human diagnostics
        "reasons": [
            _issue_to_dict(x) for x in (getattr(decision, "reasons", None) or [])
        ],
        "warnings": [
            _issue_to_dict(x) for x in (getattr(decision, "warnings", None) or [])
        ],

        # Arbitrary metrics
        "metrics": dict(getattr(decision, "metrics", None) or {}),
    }

    # Compact artifact (nulls removed)
    payload = {k: v for k, v in payload.items() if v is not None}

    validate_internal(POLICY_DECISION_SCHEMA, payload)
    return payload


def render_policy_decision_json(decision: Decision) -> str:
    import json

    payload = _decision_to_payload(decision)
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def write_policy_decision_artifact(decision: Decision, out_path: Pathish) -> Path:
    payload = _decision_to_payload(decision)
    return write_policy_decision(out_path, payload)


def write_latest_policy_decision(decision: Decision, out_dir: Pathish) -> Path:
    return write_policy_decision_artifact(decision, Path(out_dir) / "latest.json")


def load_policy_decision(path: Pathish) -> PolicyDecisionSnapshot:
    return read_policy_decision(path)


def default_policy_out_path() -> Path:
    env = os.getenv("POLICY_OUT_PATH", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return (Path("policy_out") / "latest.json").resolve()