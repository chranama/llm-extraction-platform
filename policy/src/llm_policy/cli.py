# policy/src/llm_policy/cli.py
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional, Tuple, Any

from llm_policy.config import PolicyConfig, load_extract_thresholds, load_generate_thresholds
from llm_policy.io.eval_runs import load_eval_artifact
from llm_policy.io.generate_slo import read_generate_slo_snapshot_result
from llm_policy.io.models_yaml import patch_models_yaml
from llm_policy.io.policy_decisions import default_policy_out_path, write_policy_decision_artifact
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
        prog="llm-policy",
        description="Policy engine for gating and shaping LLM capabilities.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # ------------------------------------------------------------------
    # decide-extract (now pipeline-aware)
    # ------------------------------------------------------------------

    d = sub.add_parser(
        "decide-extract",
        help="Run policy decision pipeline and emit runtime policy artifact",
    )

    d.add_argument(
        "--pipeline",
        required=True,
        choices=[
            "extract_only",
            "generate_clamp_only",
            "extract_plus_generate_clamp",
        ],
        help="Policy pipeline to execute",
    )

    d.add_argument(
        "--run-dir",
        type=str,
        default=os.getenv("POLICY_RUN_DIR", "latest"),
        help=(
            "Path to eval run directory (contains summary.json), or 'latest' "
            "(default: $POLICY_RUN_DIR or 'latest')."
        ),
    )

    d.add_argument(
        "--threshold-profile",
        type=str,
        default=None,
        help="Extract threshold profile, e.g. extract/sroie",
    )

    d.add_argument(
        "--thresholds-root",
        type=str,
        default=None,
        help="Override thresholds root directory",
    )

    # -----------------------------
    # Generate clamp (Phase 2)
    # -----------------------------

    d.add_argument(
        "--generate-threshold-profile",
        type=str,
        default=os.getenv("POLICY_GENERATE_PROFILE", "generate/portable"),
        help="Generate clamp threshold profile",
    )

    d.add_argument(
        "--no-generate-clamp",
        action="store_true",
        help="Disable generate SLO clamp enrichment (debug only)",
    )

    d.add_argument(
        "--generate-slo-path",
        type=str,
        default=os.getenv("POLICY_GENERATE_SLO_PATH", "").strip() or None,
        help="Override path to generate SLO snapshot JSON",
    )

    # -----------------------------
    # Reporting / artifacts
    # -----------------------------

    d.add_argument(
        "--report",
        type=str,
        default="text",
        choices=["text", "md"],
        help="Human report format",
    )

    d.add_argument(
        "--report-out",
        type=str,
        default=None,
        help="Write human report to file (optional)",
    )

    d.add_argument(
        "--artifact-out",
        type=str,
        default=None,
        help="Write runtime policy decision artifact JSON to this path",
    )

    d.add_argument(
        "--no-write-artifact",
        action="store_true",
        help="Do not write runtime policy artifact (debug only)",
    )

    # ------------------------------------------------------------------
    # patch-models (unchanged)
    # ------------------------------------------------------------------

    pm = sub.add_parser(
        "patch-models",
        help="Apply a decision to models.yaml by editing capabilities",
    )
    pm.add_argument("--models-yaml", type=str, required=True)
    pm.add_argument("--model-id", type=str, required=True)
    pm.add_argument("--enable-extract", action="store_true")
    pm.add_argument("--disable-extract", action="store_true")
    pm.add_argument("--dry-run", action="store_true")

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


def _render_human(decision: Decision, fmt: str) -> str:
    if fmt == "md":
        return render_decision_md(decision)
    return render_decision_text(decision)


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


# ------------------------------------------------------------------------------
# Pipeline execution
# ------------------------------------------------------------------------------


def _apply_generate_clamp(args, decision: Decision) -> Decision:
    if args.no_generate_clamp:
        return decision

    pcfg = PolicyConfig.default()
    if args.thresholds_root:
        pcfg = PolicyConfig(thresholds_root=args.thresholds_root)

    slo_res = read_generate_slo_snapshot_result(args.generate_slo_path)
    slo_ok, slo_snap, slo_path, slo_err = _unwrap_slo_read_result(slo_res)

    resolved_prof, gen_th = load_generate_thresholds(
        cfg=pcfg, profile=args.generate_threshold_profile
    )
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
                "warnings": updated.warnings
                + [
                    DecisionWarning(
                        code="generate_slo_snapshot_read_failed",
                        message="Generate SLO snapshot unreadable; safe clamp may apply",
                        context={"path": slo_path, "error": slo_err},
                    )
                ]
            }
        )

    updated = updated.model_copy(
        update={
            "reasons": updated.reasons
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


def _build_decision(args) -> Decision:
    pipeline: PipelineType = args.pipeline  # validated by argparse

    # -----------------------------
    # Base decision (fail-closed)
    # -----------------------------

    decision = Decision(
        policy="llm_policy",
        pipeline=pipeline,
    )

    # -----------------------------
    # Extract enablement
    # -----------------------------

    if pipeline in ("extract_only", "extract_plus_generate_clamp"):
        artifact = load_eval_artifact(args.run_dir or "latest")

        pcfg = PolicyConfig.default()
        if args.thresholds_root:
            pcfg = PolicyConfig(thresholds_root=args.thresholds_root)

        profile, th = load_extract_thresholds(
            cfg=pcfg, profile=args.threshold_profile
        )

        decision = decide_extract_enablement(
            artifact,
            thresholds=th,
            thresholds_profile=profile,
        ).model_copy(
            update={
                "pipeline": pipeline,
            }
        )

    # -----------------------------
    # Generate clamp
    # -----------------------------

    if pipeline in ("generate_clamp_only", "extract_plus_generate_clamp"):
        decision = _apply_generate_clamp(args, decision)

    return decision


# ------------------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.cmd == "decide-extract":
        decision = _build_decision(args)

        if not args.no_write_artifact:
            out_path = _artifact_path_or_default(args.artifact_out)
            _validate_outfile_path(out_path)
            write_policy_decision_artifact(decision, out_path)

        rendered = _render_human(decision, args.report)
        _emit(rendered, args.report_out)

        return 0 if decision.ok() else 2

    if args.cmd == "patch-models":
        if args.enable_extract and args.disable_extract:
            print("Error: choose only one of --enable-extract or --disable-extract")
            return 2

        enable = args.enable_extract and not args.disable_extract

        res = patch_models_yaml(
            path=args.models_yaml,
            model_id=args.model_id,
            capability="extract",
            enable=enable,
            write=(not args.dry_run),
        )

        print(json.dumps({"changed": res.changed, "warnings": res.warnings}, indent=2))
        return 0

    print("Unknown command")
    return 2