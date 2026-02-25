# policy/src/llm_policy/onboarding/demo.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from llm_policy.io.models_yaml_patch import PatchChange, PatchResult, patch_models_yaml_extract_capability
from llm_policy.onboarding.runner import OnboardingEvalResult, evaluate_model_onboarding


@dataclass(frozen=True)
class OnboardingApplyResult:
    """
    Result of the closed-loop onboarding apply.

    Notes:
      - `decision_ok` is derived from the Decision object (if available).
      - `patch_ok` is whether the YAML patch step succeeded.
      - `ok` is a conservative overall ok: decision_ok AND patch_ok (when patch attempted).
      - `enable_extract` may be None if decision didn't produce a boolean.
      - `changes` reflects what was modified in models.yaml (base + profiles).
    """
    ok: bool
    decision_ok: bool
    patch_ok: bool
    changed: bool
    message: str

    enable_extract: Optional[bool]
    thresholds_profile: Optional[str]
    eval_run_dir: str

    model_id: str
    models_yaml: str

    policy_name: str
    pipeline: str

    # --- New: surfaced provenance used for the capability assessment ---
    deployment_key: Optional[str] = None

    patch_profile: Optional[str] = None
    changes: Tuple[PatchChange, ...] = ()
    decision: Optional[object] = None  # keep typed Decision out of this file to avoid import cycles


def apply_model_onboarding(
    *,
    models_yaml: str | Path,
    model_id: str,
    eval_run_dir: str = "latest",
    threshold_profile: Optional[str] = None,
    thresholds_root: Optional[str] = None,
    verbose: bool = True,
    policy_name: str = "model_onboarding",
    pipeline: str = "extract_only",
) -> OnboardingApplyResult:
    """
    Closed loop:

      1) Evaluate onboarding (policy decision)
      2) Patch models.yaml capabilities.extract for base + all overlays
      3) Print patch outcome + decision summary

    This file is intentionally "demo-shaped" but does not use "demo B" terminology.
    """
    res: OnboardingEvalResult = evaluate_model_onboarding(
        eval_run_dir=eval_run_dir,
        threshold_profile=threshold_profile,
        thresholds_root=thresholds_root,
        policy_name=policy_name,
        pipeline=pipeline,
        verbose=verbose,
    )

    decision = res.decision
    decision_ok = bool(decision.ok()) if hasattr(decision, "ok") else False

    enable_extract = getattr(decision, "enable_extract", None)
    if not isinstance(enable_extract, bool):
        return OnboardingApplyResult(
            ok=False,
            decision_ok=decision_ok,
            patch_ok=False,
            changed=False,
            message="Decision did not produce a boolean enable_extract; refusing to patch models.yaml",
            enable_extract=None,
            thresholds_profile=res.thresholds_profile,
            eval_run_dir=res.eval_run_dir,
            model_id=str(model_id),
            models_yaml=str(models_yaml),
            policy_name=str(getattr(decision, "policy", policy_name) or policy_name),
            pipeline=str(getattr(decision, "pipeline", pipeline) or pipeline),
            deployment_key=res.deployment_key,
            patch_profile=res.thresholds_profile,
            changes=(),
            decision=decision,
        )

    # New: pass deployment info through to the patch step.
    patch_res: PatchResult = patch_models_yaml_extract_capability(
        Path(models_yaml),
        model_id=str(model_id),
        enable=bool(enable_extract),
        profile=res.thresholds_profile,
        deployment=res.deployment,
        deployment_key=res.deployment_key,
        assessed=True,  # flip assessed to true after an assessment
        assessed_by=str(getattr(decision, "policy", policy_name) or policy_name),
        assessed_pipeline=str(getattr(decision, "pipeline", pipeline) or pipeline),
        eval_run_dir=str(res.eval_run_dir),
        thresholds_profile=str(res.thresholds_profile) if res.thresholds_profile is not None else None,
    )

    if verbose:
        print("\n" + "=" * 80)
        print("MODEL ONBOARDING — APPLY")
        print("=" * 80)
        print(f"models_yaml: {str(models_yaml)}")
        print(f"model_id: {model_id}")
        print(f"enable_extract: {enable_extract}")
        print(f"decision_ok: {decision_ok}")
        if res.deployment_key:
            print(f"deployment_key: {res.deployment_key}")
        print(f"patch: ok={patch_res.ok} changed={patch_res.changed} msg={patch_res.message}")
        if patch_res.changes:
            print("changes:")
            for ch in patch_res.changes:
                # ch.after is now a small dict snapshot; keep output readable
                print(f"  - {ch.scope}: capabilities.extract {ch.before!r} -> {ch.after!r}")
        else:
            print("changes: (none)")
        print("=" * 80 + "\n")

    patch_ok = bool(patch_res.ok)
    overall_ok = bool(decision_ok and patch_ok)

    return OnboardingApplyResult(
        ok=overall_ok,
        decision_ok=decision_ok,
        patch_ok=patch_ok,
        changed=bool(patch_res.changed),
        message=str(patch_res.message),
        enable_extract=bool(enable_extract),
        thresholds_profile=res.thresholds_profile,
        eval_run_dir=res.eval_run_dir,
        model_id=str(model_id),
        models_yaml=str(models_yaml),
        policy_name=str(getattr(decision, "policy", policy_name) or policy_name),
        pipeline=str(getattr(decision, "pipeline", pipeline) or pipeline),
        deployment_key=res.deployment_key,
        patch_profile=res.thresholds_profile,
        changes=tuple(patch_res.changes or ()),
        decision=decision,
    )