# policy/src/llm_policy/onboarding/runner.py
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

from llm_policy.config import PolicyConfig, load_extract_thresholds
from llm_policy.io.eval_runs import load_eval_artifact
from llm_policy.policies.extract_enablement import decide_extract_enablement
from llm_policy.types.decision import Decision
from llm_policy.types.eval_artifact import EvalArtifact


@dataclass(frozen=True)
class OnboardingEvalResult:
    """
    Generalized result object for onboarding evaluation.

    This is intentionally policy-agnostic but currently supports extract enablement.
    """
    decision: Decision
    eval_run_dir: str
    thresholds_profile: Optional[str]

    # --- New: deployment provenance carried forward into capability determination ---
    deployment: Optional[Dict[str, Any]] = None
    deployment_key: Optional[str] = None

    model_id: Optional[str] = None
    notes: Optional[str] = None


def _short_ctx(ctx: Any, *, max_len: int = 800) -> str:
    if not isinstance(ctx, dict) or not ctx:
        return ""
    try:
        s = json.dumps(ctx, ensure_ascii=False, sort_keys=True)
    except Exception:
        s = str(ctx)
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def _print_decision(decision: Decision) -> None:
    enable = getattr(decision, "enable_extract", None)
    status = str(getattr(decision, "status", "") or "")
    pipeline = str(getattr(decision, "pipeline", "") or "")

    print("\n" + "=" * 80)
    print("MODEL ONBOARDING — EVALUATION RESULT")
    print("=" * 80)
    print(f"pipeline: {pipeline}")
    print(f"status:   {status}")
    print(f"enable_extract: {enable!r}")

    if decision.reasons:
        print("\nreasons:")
        for r in decision.reasons:
            ctx = _short_ctx(getattr(r, "context", None))
            suffix = f"  ctx={ctx}" if ctx else ""
            print(f"  - [{r.code}] {r.message}{suffix}")

    if decision.warnings:
        print("\nwarnings:")
        for w in decision.warnings:
            ctx = _short_ctx(getattr(w, "context", None))
            suffix = f"  ctx={ctx}" if ctx else ""
            print(f"  - [{w.code}] {w.message}{suffix}")

    # Show deployment provenance if present (pulled from decision.metrics)
    try:
        metrics = getattr(decision, "metrics", None) or {}
        if isinstance(metrics, dict):
            dep = metrics.get("deployment")
            dep_key = metrics.get("deployment_key")
            if dep is not None or dep_key is not None:
                print("\nprovenance:")
                if dep_key is not None:
                    print(f"  - deployment_key: {dep_key}")
                if dep is not None:
                    print(f"  - deployment: {_short_ctx(dep, max_len=1200) or str(dep)}")
    except Exception:
        pass

    print("=" * 80 + "\n")


def _deployment_from_artifact(artifact: EvalArtifact) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Pull deployment provenance from eval summary.

    Assumes your eval summary contract now includes:
      - deployment: object (recommended)
      - deployment_key: string (recommended)
    """
    s = getattr(artifact, "summary", None)
    if s is None:
        return None, None

    dep = getattr(s, "deployment", None)
    dep_key = getattr(s, "deployment_key", None)

    dep_dict: Optional[Dict[str, Any]] = None
    if isinstance(dep, dict):
        dep_dict = dict(dep)

    dep_key_str: Optional[str] = None
    if isinstance(dep_key, str) and dep_key.strip():
        dep_key_str = dep_key.strip()

    return dep_dict, dep_key_str


def evaluate_model_onboarding(
    *,
    eval_run_dir: str = "latest",
    threshold_profile: Optional[str] = None,
    thresholds_root: Optional[str] = None,
    policy_name: str = "model_onboarding",
    pipeline: str = "extract_only",
    verbose: bool = True,
) -> OnboardingEvalResult:
    """
    Generalized onboarding evaluator.

    Today:
      - loads eval artifact (summary/results)
      - runs extract enablement policy (Phase 2 hardened)
      - returns a Decision for downstream orchestration

    Future:
      - add more policies (e.g., different extraction tasks, additional gates) without changing CLI wiring.
    """
    artifact: EvalArtifact = load_eval_artifact(eval_run_dir or "latest")

    pcfg = PolicyConfig.default()
    if thresholds_root:
        pcfg = PolicyConfig(thresholds_root=thresholds_root)

    prof, th = load_extract_thresholds(cfg=pcfg, profile=threshold_profile)

    decision = decide_extract_enablement(
        artifact,
        thresholds=th,
        thresholds_profile=prof,
    ).model_copy(update={"pipeline": pipeline, "policy": policy_name})

    # Pull deployment info from eval summary and ALSO stamp into decision.metrics for visibility.
    dep, dep_key = _deployment_from_artifact(artifact)
    try:
        metrics = dict(getattr(decision, "metrics", None) or {})
        if dep is not None:
            metrics["deployment"] = dep
        if dep_key is not None:
            metrics["deployment_key"] = dep_key
        decision = decision.model_copy(update={"metrics": metrics})
    except Exception:
        pass

    if verbose:
        _print_decision(decision)

    return OnboardingEvalResult(
        decision=decision,
        eval_run_dir=str(eval_run_dir),
        thresholds_profile=str(prof) if prof is not None else None,
        deployment=dep,
        deployment_key=dep_key,
    )