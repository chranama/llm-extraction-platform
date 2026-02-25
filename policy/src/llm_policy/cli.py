# policy/src/llm_policy/cli.py
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Tuple

from llm_contracts.schema import validate_internal  # type: ignore

from llm_policy.config import PolicyConfig, load_extract_thresholds, load_generate_thresholds
from llm_policy.io.eval_runs import load_eval_artifact
from llm_policy.io.generate_slo import read_generate_slo_snapshot_result
from llm_policy.io.policy_decisions import (
    POLICY_DECISION_SCHEMA,
    default_policy_out_path,
    write_policy_decision_artifact,
)
from llm_policy.onboarding.demo import apply_model_onboarding
from llm_policy.onboarding.runner import evaluate_model_onboarding
from llm_policy.policies.extract_enablement import decide_extract_enablement
from llm_policy.policies.generate_slo_clamp import decide_generate_slo_clamp, thresholds_from_mapping
from llm_policy.reports.writer import render_decision_md, render_decision_text
from llm_policy.types.decision import (
    Decision,
    DecisionReason,
    DecisionWarning,
    PipelineType,
)

# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="policy",
        description="Policy engine for gating and shaping LLM capabilities.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # ------------------------------------------------------------------
    # runtime-decision (runtime policy artifact, optional generate clamp)
    # ------------------------------------------------------------------
    rd = sub.add_parser(
        "runtime-decision",
        help="Compute runtime policy decision and write policy_decision_v2 artifact.",
    )

    rd.add_argument(
        "--pipeline",
        required=True,
        choices=[
            "extract_only",
            "generate_clamp_only",
            "extract_plus_generate_clamp",
        ],
        help="Policy pipeline to execute.",
    )

    rd.add_argument(
        "--run-dir",
        type=str,
        default=os.getenv("POLICY_RUN_DIR", "latest"),
        help=(
            "Path to eval run directory (contains summary.json), or 'latest' "
            "(default: $POLICY_RUN_DIR or 'latest')."
        ),
    )

    rd.add_argument(
        "--threshold-profile",
        type=str,
        default=None,
        help="Extract threshold profile, e.g. extract/sroie (optional).",
    )

    rd.add_argument(
        "--thresholds-root",
        type=str,
        default=None,
        help="Override thresholds root directory (optional).",
    )

    # Generate clamp wiring (triggered only if pipeline includes generate clamp)
    rd.add_argument(
        "--generate-threshold-profile",
        type=str,
        default=os.getenv("POLICY_GENERATE_PROFILE", "generate/portable"),
        help="Generate clamp threshold profile.",
    )

    rd.add_argument(
        "--no-generate-clamp",
        action="store_true",
        help="Disable generate SLO clamp enrichment (debug only).",
    )

    rd.add_argument(
        "--generate-slo-path",
        type=str,
        default=os.getenv("POLICY_GENERATE_SLO_PATH", "").strip() or None,
        help="Override path to generate SLO snapshot JSON.",
    )

    # Reporting / artifacts
    rd.add_argument(
        "--report",
        type=str,
        default="text",
        choices=["text", "md"],
        help="Human report format.",
    )

    rd.add_argument(
        "--report-out",
        type=str,
        default=None,
        help="Write human report to file (optional).",
    )

    rd.add_argument(
        "--artifact-out",
        type=str,
        default=None,
        help="Write runtime policy decision artifact JSON to this path (optional).",
    )

    rd.add_argument(
        "--no-write-artifact",
        action="store_true",
        help="Do not write runtime policy artifact (debug only).",
    )

    # ------------------------------------------------------------------
    # model-onboarding (closed loop: evaluate + patch models.yaml)
    # ------------------------------------------------------------------
    mo = sub.add_parser(
        "model-onboarding",
        help="Evaluate onboarding and (optionally) patch models.yaml capabilities.extract.",
    )

    mo_sub = mo.add_subparsers(dest="subcmd", required=True)

    mo_eval = mo_sub.add_parser("evaluate", help="Evaluate onboarding policy against an eval run dir.")
    mo_eval.add_argument(
        "--eval-run-dir",
        type=str,
        default=os.getenv("POLICY_RUN_DIR", "latest"),
        help="Eval run directory, or 'latest' (default: $POLICY_RUN_DIR or 'latest').",
    )
    mo_eval.add_argument(
        "--threshold-profile",
        type=str,
        default=None,
        help="Extract threshold profile, e.g. extract/sroie (optional).",
    )
    mo_eval.add_argument(
        "--thresholds-root",
        type=str,
        default=None,
        help="Override thresholds root directory (optional).",
    )
    mo_eval.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress pretty printed decision output.",
    )

    mo_apply = mo_sub.add_parser("apply", help="Evaluate onboarding and patch models.yaml.")
    mo_apply.add_argument(
        "--models-yaml",
        type=str,
        required=True,
        help="Path to config/models.yaml to patch.",
    )
    mo_apply.add_argument(
        "--model-id",
        type=str,
        required=True,
        help="Model id to patch in base and all profiles.",
    )
    mo_apply.add_argument(
        "--eval-run-dir",
        type=str,
        default=os.getenv("POLICY_RUN_DIR", "latest"),
        help="Eval run directory, or 'latest' (default: $POLICY_RUN_DIR or 'latest').",
    )
    mo_apply.add_argument(
        "--threshold-profile",
        type=str,
        default=None,
        help="Extract threshold profile, e.g. extract/sroie (optional).",
    )
    mo_apply.add_argument(
        "--thresholds-root",
        type=str,
        default=None,
        help="Override thresholds root directory (optional).",
    )
    mo_apply.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress pretty printed decision + patch output.",
    )

    return p


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------


def _emit(s: str, out: Optional[str]) -> None:
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write(s)
    else:
        print(s, end="")


def _provenance_prefix(decision: Decision) -> str:
    """
    Render a short provenance header so deployment context is visible immediately.

    We keep this CLI-owned (not in writer.py) to avoid changing existing report formatting.
    """
    pipeline = str(getattr(decision, "pipeline", "") or "")
    if pipeline not in ("extract_only", "extract_plus_generate_clamp"):
        return ""

    m = getattr(decision, "metrics", None) or {}
    if not isinstance(m, dict):
        return ""

    dep_key = m.get("deployment_key")
    dep = m.get("deployment")
    eval_run_dir = getattr(decision, "eval_run_dir", None)

    # Keep it short & stable.
    lines = [
        "== Provenance ==",
        f"eval_run_dir: {str(eval_run_dir or '')}",
        f"deployment_key: {str(dep_key or '')}",
    ]

    # deployment is often a dict; show as compact JSON on one line.
    if dep is not None:
        try:
            dep_s = json.dumps(dep, ensure_ascii=False, sort_keys=True)
        except Exception:
            dep_s = str(dep)
        lines.append(f"deployment: {dep_s}")

    lines.append("")  # spacer
    return "\n".join(lines)


def _render_human(decision: Decision, fmt: str) -> str:
    # For generate-only, "enable_extract" is not meaningful; display as None
    # even if the internal Decision model defaults it to False.
    if str(getattr(decision, "pipeline", "") or "") == "generate_clamp_only":
        try:
            decision_for_report = decision.model_copy(update={"enable_extract": None})
        except Exception:
            decision_for_report = decision
    else:
        decision_for_report = decision

    prefix = _provenance_prefix(decision_for_report)

    if fmt == "md":
        return prefix + render_decision_md(decision_for_report)
    return prefix + render_decision_text(decision_for_report)


def _artifact_path_or_default(raw: Optional[str]) -> str:
    if raw and raw.strip():
        return raw.strip()
    env = os.getenv("POLICY_OUT_PATH", "").strip()
    if env:
        return env
    return str(default_policy_out_path())


def _validate_outfile_path(p: str) -> None:
    pp = Path(p)
    if pp.name in ("", ".", ".."):
        raise ValueError(f"artifact path must be a file path, got: {p!r}")


def _unwrap_slo_read_result(res: Any) -> Tuple[bool, Any, str, Optional[str]]:
    ok = bool(getattr(res, "ok", False))
    snap = getattr(res, "artifact", None) or getattr(res, "snapshot", None)
    path = getattr(res, "resolved_path", None) or getattr(res, "path", "") or ""
    err = getattr(res, "error", None)
    return ok, snap, str(path), str(err) if err is not None else None


def _is_generate_only(decision: Decision) -> bool:
    pipeline = str(getattr(decision, "pipeline", "") or "")
    return pipeline == "generate_clamp_only"


def _normalize_generate_only(decision: Decision) -> Decision:
    """
    generate_clamp_only is SHAPING ONLY (never gating).
    Contract requirements for policy_decision_v2:
      - status = "allow"
      - ok = true
      - enable_extract = null
    """
    if not _is_generate_only(decision):
        return decision
    decision = decision.model_copy(update={"status": "allow"})
    return decision


def _write_generate_only_artifact(decision: Decision, out_path: str) -> None:
    """
    Write a policy_decision_v2 artifact for generate_clamp_only while forcibly
    satisfying the contract: enable_extract must be null.
    """
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    reasons = [r.model_dump() if hasattr(r, "model_dump") else dict(r) for r in (decision.reasons or [])]
    warnings = [w.model_dump() if hasattr(w, "model_dump") else dict(w) for w in (decision.warnings or [])]

    payload = {
        "schema_version": "policy_decision_v2",
        "generated_at": now,
        "policy": getattr(decision, "policy", "llm_policy"),
        "pipeline": getattr(decision, "pipeline", "generate_clamp_only"),
        "status": "allow",
        "ok": True,
        "enable_extract": None,  # contract: must be null in generate_clamp_only
        "generate_max_new_tokens_cap": getattr(decision, "generate_max_new_tokens_cap", None),
        "contract_errors": int(getattr(decision, "contract_errors", 0) or 0),
        "contract_warnings": int(getattr(decision, "contract_warnings", 0) or 0),
        "thresholds_profile": getattr(decision, "thresholds_profile", None),
        "generate_thresholds_profile": getattr(decision, "generate_thresholds_profile", None),
        "eval_run_dir": getattr(decision, "eval_run_dir", None),
        "reasons": reasons,
        "warnings": warnings,
        "metrics": getattr(decision, "metrics", {}) or {},
    }

    validate_internal(POLICY_DECISION_SCHEMA, payload)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=False)


# ------------------------------------------------------------------------------
# Pipeline execution (runtime-decision)
# ------------------------------------------------------------------------------


def _apply_generate_clamp(args, decision: Decision) -> Decision:
    if args.no_generate_clamp:
        return decision

    pcfg = PolicyConfig.default()
    if args.thresholds_root:
        pcfg = PolicyConfig(thresholds_root=args.thresholds_root)

    slo_res = read_generate_slo_snapshot_result(args.generate_slo_path)
    slo_ok, slo_snap, slo_path, slo_err = _unwrap_slo_read_result(slo_res)

    resolved_prof, gen_th = load_generate_thresholds(cfg=pcfg, profile=args.generate_threshold_profile)
    th = thresholds_from_mapping(gen_th.model_dump())

    updated = decide_generate_slo_clamp(
        base=decision,
        slo=slo_snap,
        thresholds=th,
        policy_name="generate_slo_clamp",
        enabled=True,
    )

    if not slo_ok:
        updated = updated.model_copy(
            update={
                "warnings": list(updated.warnings)
                + [
                    DecisionWarning(
                        code="generate_slo_snapshot_read_failed",
                        message="Generate SLO snapshot unreadable; no clamp applied (Option A).",
                        context={"path": slo_path, "error": slo_err},
                    )
                ]
            }
        )

    updated = updated.model_copy(
        update={
            "reasons": list(updated.reasons)
            + [
                DecisionReason(
                    code="generate_slo_profile",
                    message="Generate SLO clamp thresholds applied",
                    context={"profile": resolved_prof},
                )
            ],
            "generate_thresholds_profile": resolved_prof,
        }
    )

    return updated


def _build_runtime_decision(args) -> Decision:
    pipeline: PipelineType = args.pipeline  # validated by argparse

    decision = Decision(policy="llm_policy", pipeline=pipeline)

    # Extract enablement
    if pipeline in ("extract_only", "extract_plus_generate_clamp"):
        artifact = load_eval_artifact(args.run_dir or "latest")

        pcfg = PolicyConfig.default()
        if args.thresholds_root:
            pcfg = PolicyConfig(thresholds_root=args.thresholds_root)

        prof, th = load_extract_thresholds(cfg=pcfg, profile=args.threshold_profile)
        decision = decide_extract_enablement(
            artifact,
            thresholds=th,
            thresholds_profile=prof,
        ).model_copy(update={"pipeline": pipeline})

    # Generate clamp
    if pipeline in ("generate_clamp_only", "extract_plus_generate_clamp"):
        decision = _apply_generate_clamp(args, decision)

    decision = _normalize_generate_only(decision)
    return decision


def _write_artifact(decision: Decision, out_path: str) -> None:
    if _is_generate_only(decision):
        _write_generate_only_artifact(decision, out_path)
        return
    write_policy_decision_artifact(decision, out_path)


# ------------------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # ------------------------
    # runtime-decision
    # ------------------------
    if args.cmd == "runtime-decision":
        decision = _build_runtime_decision(args)

        if not args.no_write_artifact:
            out_path = _artifact_path_or_default(args.artifact_out)
            _validate_outfile_path(out_path)
            _write_artifact(decision, out_path)

        rendered = _render_human(decision, args.report)
        _emit(rendered, args.report_out)

        return 0 if _is_generate_only(decision) or decision.ok() else 2

    # ------------------------
    # model-onboarding
    # ------------------------
    if args.cmd == "model-onboarding":
        if args.subcmd == "evaluate":
            res = evaluate_model_onboarding(
                eval_run_dir=args.eval_run_dir,
                threshold_profile=args.threshold_profile,
                thresholds_root=args.thresholds_root,
                policy_name="model_onboarding",
                pipeline="extract_only",
                verbose=not bool(args.quiet),
            )
            # Align exit code with Decision.ok() semantics
            return 0 if res.decision.ok() else 2

        if args.subcmd == "apply":
            apply_res = apply_model_onboarding(
                models_yaml=args.models_yaml,
                model_id=args.model_id,
                eval_run_dir=args.eval_run_dir,
                threshold_profile=args.threshold_profile,
                thresholds_root=args.thresholds_root,
                verbose=not bool(args.quiet),
            )
            return 0 if bool(apply_res.ok) else 2

        print("Unknown model-onboarding subcommand", flush=True)
        return 2

    print("Unknown command", flush=True)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())